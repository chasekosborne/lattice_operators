"""Write projected operators to HDF5 in the OpPy (OperatorConst) layout.

Every file follows the layout of ``construct_baryon_ops.py`` from OpPy::

  /<root>/<channel>/<irrep>_<row>/<prefix>_<n>/
      OperatorInfo : str
      ivals        : [n_terms, n_ints, <n_ints integers per term>...]
      dvals        : [re_0, im_0, re_1, im_1, ...]

Each term is a product of color-contracted hadron elementals.  A hadron with
quark slots ``f_1 ... f_k`` (Dirac indices ``s``, 1-based Dirac-Pauli, and
displacements ``d``, 0 or +-1, +-2, +-3 for +-x, +-y, +-z) is

  baryon       eps_{abc} f1^a f2^b f3^c
  meson        delta_{ab} fbar1^a f2^b                  (antiquark slot first)
  tetraquark   eps_{abc} eps_{ade} f1^b f2^c fbar3^d fbar4^e

Coefficients are summed over the slot permutations that leave the flavors
unchanged, as in OpPy's symmetrized output.

Single-hadron operators (``prefix = SS``) store ``s_1..s_k, d_1..d_k`` per term;
for single baryons this is exactly the OpPy format, with ``<channel>`` the OpPy
flavor class (``S3+``, ``Sigma1+``, ..., ``E``).  Multi-hadron operators
(``prefix = MH``) store, for every hadron block of a term,
``px, py, pz, f_1..f_k, s_1..s_k, d_1..d_k`` where ``f`` indexes the
``flavor_table`` attribute.  Each operator group carries a ``layout``
attribute spelling out its integers.
"""

import itertools
import math
from collections import defaultdict
from fractions import Fraction

import h5py
import numpy as np
from sympy import Mul, N

from .cubic_rotations import E
from .grassmann import GrassmannSymbol
from .operators import canonical_product_term, parse_quark_symbol

FLOAT_TOL = 1e-10
SINGLE_HADRON_PREFIX = "SS"
MULTI_HADRON_PREFIX = "MH"
DISPLACEMENT_CLASS = SINGLE_HADRON_PREFIX

_QUARK_ORDER = "udscbt"

# OpPy's H irrep basis differs from the one in spinor_irreps by these row phases.
OPPY_ROW_PHASES = {
    "Hg": {2: -1, 4: -1},
    "Hu": {2: -1, 4: -1},
}


def _permutation_sign(perm):
  sign = 1
  perm = list(perm)
  for i in range(len(perm)):
    while perm[i] != i:
      j = perm[i]
      perm[i], perm[j] = perm[j], perm[i]
      sign = -sign
  return sign


def _sorted_with_sign(names):
  """Sort Grassmann symbol names, returning the sign of the reordering."""
  order = sorted(range(len(names)), key=lambda i: names[i])
  return tuple(names[i] for i in order), _permutation_sign(order)


def _field_name(flavor, color, dirac):
  return "{}_{{{},{}}}".format(flavor, color, dirac)


def _is_anti(flavor):
  return flavor.endswith("+")


def readable_flavor(flavor):
  """``"u+"`` (the internal antiquark name) -> ``"ubar"``."""
  return flavor[:-1] + "bar" if _is_anti(flavor) else flavor


def _flavor_rank(flavor):
  base = flavor.rstrip("+")
  return (_QUARK_ORDER.index(base) if base in _QUARK_ORDER else len(_QUARK_ORDER), base)


def default_flavor_order(flavors):
  """Slot order for a flavor multiset: repeated flavors first, then ``udscbt`` order.

  Gives ``uuu``, ``uud``, ``uus``, ``uds``, i.e. the OpPy reference orderings.
  """
  counts = {f: list(flavors).count(f) for f in set(flavors)}
  return tuple(sorted(flavors, key=lambda f: (-counts[f], _flavor_rank(f))))


def flavor_class(flavors):
  """OpPy symmetry class of a three-quark slot flavor assignment."""
  f1, f2, f3 = flavors
  if f1 == f2 == f3:
    return "S3+"
  if f1 == f2:
    return "Sigma1+"
  if f1 == f3:
    return "Sigma2+"
  if f2 == f3:
    return "Sigma3+"
  return "E"


# # # # Hadron color structures # # # #

def _eps(colors):
  return _permutation_sign(colors) if sorted(colors) == [0, 1, 2] else 0


def _delta(colors):
  return 1 if colors[0] == colors[1] else 0


def _eps_eps(colors):
  # sum_a eps_{a c1 c2} eps_{a c3 c4} = delta_{c1 c3} delta_{c2 c4} - delta_{c1 c4} delta_{c2 c3}
  c1, c2, c3, c4 = colors
  if c1 == c2:
    return 0
  if (c3, c4) == (c1, c2):
    return 1
  if (c3, c4) == (c2, c1):
    return -1
  return 0


# (number of quarks, number of antiquarks) -> (name, color tensor, reference colors per slot)
_HADRON_TYPES = {
    (3, 0): ("baryon", _eps, (0, 1, 2)),
    (0, 3): ("antibaryon", _eps, (0, 1, 2)),
    (1, 1): ("meson", _delta, (0, 0)),
    (2, 2): ("tetraquark", _eps_eps, (0, 1, 0, 1)),
}


def _hadron_type(slots):
  key = (sum(not _is_anti(f) for f in slots), sum(_is_anti(f) for f in slots))
  try:
    return _HADRON_TYPES[key]
  except KeyError:
    raise TypeError("No color-contracted elemental for a hadron with {} quarks and {} "
                    "antiquarks".format(*key))


def slot_order(flavors):
  """Quark slot order of one hadron: quarks then antiquarks, antiquark first for mesons."""
  quarks = default_flavor_order([f for f in flavors if not _is_anti(f)])
  antiquarks = default_flavor_order([f for f in flavors if _is_anti(f)])
  if len(quarks) == 1 and len(antiquarks) == 1:
    return antiquarks + quarks
  return quarks + antiquarks


def _flavor_stabilizer(slots):
  return [p for p in itertools.permutations(range(len(slots)))
          if all(slots[p[i]] == slots[i] for i in range(len(slots)))]


def _spin_stabilizer_size(slots, spins):
  return sum(1 for p in _flavor_stabilizer(slots)
             if all(spins[p[i]] == spins[i] for i in range(len(slots))))


# # # # Grassmann coefficients -> elementals # # # #

def _term_blocks(term):
  if isinstance(term, tuple) and len(term) == 2 and not isinstance(term[0], tuple):
    return [term]
  return list(term)


def _momentum_key(momentum):
  if momentum is None:
    return None
  return (int(momentum[0]), int(momentum[1]), int(momentum[2]))


def _block_elemental(monomial, slots=None):
  """``(slots, spins, sign)`` if the block's fields sit at the reference colors of
  their slots, else ``(slots, None, 0)``; ``coefficient * sign`` is the coefficient
  of the slot-ordered product."""
  factors = monomial.args if isinstance(monomial, Mul) else (monomial,)
  fields = [parse_quark_symbol(symbol)[:3] for symbol in factors]
  names = [f[0] for f in fields]
  if slots is None:
    slots = slot_order(names)
  elif sorted(slots) != sorted(names):
    raise ValueError("Slot flavors {} do not match quark content {}".format(
        tuple(slots), tuple(sorted(names))))
  _, _, reference = _hadron_type(slots)

  slot_of = {(_is_anti(f), col): i for i, (f, col) in enumerate(zip(slots, reference))}
  assigned = [None] * len(slots)
  for position, (flavor, color, dirac) in enumerate(fields):
    i = slot_of.get((_is_anti(flavor), color))
    if i is None or assigned[i] is not None or slots[i] != flavor:
      return slots, None, 0
    assigned[i] = (position, dirac)
  sign = _permutation_sign([a[0] for a in assigned])
  return slots, tuple(a[1] + 1 for a in assigned), sign


def operator_elementals(coefficients, flavors=None, check=True):
  """Express an operator's Grassmann coefficients as products of hadron elementals.

  Parameters
  ----------
  coefficients : dict
      ``.coefficients`` of an ``Operator``, ``OperatorAdd`` or ``OperatorMul``.
  flavors : sequence[str] | None
      Slot flavors of a single-hadron operator, e.g. ``"uud"``; defaults to
      ``slot_order`` of each hadron's content.
  check : bool
      Rebuild the operator from the elementals and raise if it differs.

  Returns
  -------
  dict
      ``{((momentum, slots, spins), ...): coefficient}`` with one entry per
      hadron block, in the canonical block order of the operator's terms.
  """
  if isinstance(flavors, str):
    flavors = tuple(flavors)
  entries = defaultdict(complex)
  momenta = dict()
  numeric = dict()
  n_blocks = None
  for term, coeff in coefficients.items():
    c = complex(N(coeff))
    if abs(c) < FLOAT_TOL:
      continue
    numeric[term] = c
    blocks = _term_blocks(term)
    if n_blocks is None:
      n_blocks = len(blocks)
    elif n_blocks != len(blocks):
      raise NotImplementedError("Operators mixing single- and multi-hadron terms cannot be "
                                "written; write each part separately")
    keys, sign = [], 1
    for momentum, monomial in blocks:
      slots, spins, block_sign = _block_elemental(monomial, flavors if n_blocks == 1 else None)
      if spins is None:
        break
      key = _momentum_key(momentum)
      momenta[key] = momentum
      keys.append((key, slots, spins))
      sign *= block_sign
    else:
      entries[tuple(keys)] += sign * c

  if not numeric:
    raise ValueError("Cannot write a vanishing operator")
  entries = {k: v for k, v in entries.items() if abs(v) > FLOAT_TOL}
  if check:
    _check_reconstruction(entries, numeric, momenta)
  return entries


def _expand_block(key, momenta, cache):
  if key not in cache:
    momentum_key, slots, spins = key
    _, tensor, _ = _hadron_type(slots)
    out = defaultdict(int)
    for colors in itertools.product(range(3), repeat=len(slots)):
      t = tensor(colors)
      if not t:
        continue
      names = [_field_name(f, col, s - 1) for f, col, s in zip(slots, colors, spins)]
      if len(set(names)) < len(names):
        continue
      ordered, sign = _sorted_with_sign(names)
      block = (momenta[momentum_key], Mul(*[GrassmannSymbol(n) for n in ordered]))
      out[block] += t * sign
    cache[key] = list(out.items())
  return cache[key]


def _check_reconstruction(entries, numeric, momenta):
  rebuilt = defaultdict(complex)
  cache = dict()
  for keys, c in entries.items():
    weight = c / math.prod(len(_flavor_stabilizer(slots)) for _, slots, _ in keys)
    expansions = [_expand_block(key, momenta, cache) for key in keys]
    if len(keys) == 1:
      for block, x in expansions[0]:
        rebuilt[block] += weight * x
      continue
    for combo in itertools.product(*expansions):
      canonical, sign = canonical_product_term(tuple(block for block, _ in combo))
      if canonical is None:
        continue
      rebuilt[canonical] += weight * sign * math.prod(x for _, x in combo)

  for term in set(rebuilt) | set(numeric):
    if abs(rebuilt.get(term, 0) - numeric.get(term, 0)) > 1e-8:
      raise ValueError("Operator is not a sum of products of color-contracted hadron "
                       "elementals; cannot write it in elemental form")


def elemental_coefficients(coefficients, flavors=None):
  """Single-hadron form of ``operator_elementals``: ``(slot flavors, {spins: coeff})``."""
  entries = operator_elementals(coefficients, flavors)
  slots = {keys[0][1] for keys in entries}
  if any(len(keys) != 1 for keys in entries) or len(slots) != 1:
    raise ValueError("All terms of a single-hadron operator must have the same quark content")
  return slots.pop(), {keys[0][2]: c for keys, c in entries.items()}


# # # # Projection # # # #

def _is_single_baryon(operator):
  term = next(iter(operator.coefficients))
  blocks = _term_blocks(term)
  if len(blocks) != 1:
    return False
  monomial = blocks[0][1]
  factors = monomial.args if isinstance(monomial, Mul) else (monomial,)
  return len(factors) == 3 and not any(parse_quark_symbol(f)[3] for f in factors)


def _basis_sort_key(operator):
  """Smallest elemental Dirac key of a basis operator: OpPy's rising basis order."""
  _, elementals = elemental_coefficients(operator.coefficients)
  return min(elementals)


def oppy_row1_coefficient_rows(op_rep, irrep, irrep_matrices, use_generators=False):
  """Row-1 operators chosen as OpPy does: projections of the basis operators,
  taken in rising Dirac order, keeping each one that is linearly independent."""
  P = op_rep.getProjectionMatrix(
      irrep, row=1, irrep_matrices=irrep_matrices, use_generators=use_generators
  )
  P_num = np.array(P.evalf(), dtype=complex)
  rank = np.linalg.matrix_rank(P_num, tol=1e-8)
  basis_ops = list(op_rep.basis.operators)
  order = sorted(range(len(basis_ops)), key=lambda n: _basis_sort_key(basis_ops[n]))

  selected = []
  for n in order:
    if len(selected) == rank:
      break
    trial = P_num[selected + [n], :]
    if np.linalg.matrix_rank(trial, tol=1e-8) == len(selected) + 1:
      selected.append(n)
  return [P.row(n) for n in selected]


def projected_operator_rows(op_rep, irrep, irrep_matrices, use_generators=False,
                            convention="oppy"):
  """Coefficient rows ``{row: [multiple_0, multiple_1, ...]}`` in the basis of ``op_rep``.

  Row 1 is projected directly; other rows are its partners, so the same
  multiple index labels one irrep multiplet across all rows.

  ``convention="oppy"`` uses OpPy's irrep row phases and, for single baryons,
  picks row-1 operators like OpPy (see ``oppy_row1_coefficient_rows``);
  ``"native"`` uses ``getLinearlyIndependentProjectedCoefficientRows`` and the
  irrep matrices of this package unchanged.
  """
  if convention not in ("oppy", "native"):
    raise ValueError("convention must be 'oppy' or 'native'")
  dim = int(op_rep.little_group.getCharacter(irrep, E))
  basis_ops = list(op_rep.basis.operators)
  if convention == "oppy" and all(_is_single_baryon(op) for op in basis_ops):
    first = oppy_row1_coefficient_rows(op_rep, irrep, irrep_matrices, use_generators)
  else:
    first = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        irrep, row=1, irrep_matrices=irrep_matrices, use_generators=use_generators
    )
  phases = OPPY_ROW_PHASES.get(irrep, {}) if convention == "oppy" else {}
  rows = {1: [list(r) for r in first]}
  for row in range(2, dim + 1):
    partners = op_rep.getPartnerRowCoefficientRows(
        irrep, source_rows=first, target_row=row, source_row=1,
        irrep_matrices=irrep_matrices, use_generators=use_generators,
    )
    phase = phases.get(row, 1)
    rows[row] = [[phase * c for c in r] for r in partners]
  return rows


def normalize_multiplet(rows, max_denominator=1000):
  """Rescale one multiplet (all rows together) as OpPy's ``prettify_basis`` does.

  OpPy expands operators in basis elementals (one per orbit of Dirac indices
  under the flavor stabilizer) and scales by ``sqrt(lcm(q) / gcd(p))`` where the
  squared real and imaginary parts of those coefficients are ``p / q``.
  ``rows`` are ``operator_elementals`` dictionaries.
  """
  nums, denoms = [], []
  for entries in rows:
    for keys, coeff in entries.items():
      basis_coeff = coeff / math.prod(_spin_stabilizer_size(slots, spins)
                                      for _, slots, spins in keys)
      for part in (basis_coeff.real, basis_coeff.imag):
        frac = Fraction(part ** 2).limit_denominator(max_denominator)
        if abs(float(frac) - part ** 2) > 1e-8:
          return rows
        nums.append(frac.numerator)
        denoms.append(frac.denominator)
  num_gcd = math.gcd(*nums)
  if num_gcd == 0:
    return rows
  scale = math.sqrt(math.lcm(*denoms) / num_gcd)
  return [{keys: coeff * scale for keys, coeff in entries.items()} for entries in rows]


def _combine_coefficients(basis_ops, coeff_row):
  combined = dict()
  for coeff, op in zip(coeff_row, basis_ops):
    if coeff == 0:
      continue
    for term, c in op.coefficients.items():
      combined[term] = combined.get(term, 0) + coeff * c
  return combined


# # # # Writing # # # #

def _block_label(slots):
  return "".join(readable_flavor(f) for f in slots)


def _default_root(entries):
  keys = next(iter(entries))
  if len(keys) == 1:
    return "{}_operators".format(_hadron_type(keys[0][1])[0])
  names = tuple(_hadron_type(slots)[0] for _, slots, _ in keys)
  if all(n == "baryon" for n in names) and len(names) in (2, 3):
    return {2: "dibaryon_operators", 3: "tribaryon_operators"}[len(names)]
  if names == ("meson", "meson"):
    return "meson_meson_operators"
  return "multi_hadron_operators"


def _is_single_baryon_entries(entries):
  keys = next(iter(entries))
  return len(keys) == 1 and _hadron_type(keys[0][1])[0] == "baryon"


def _default_channel(entries):
  keys = next(iter(entries))
  if _is_single_baryon_entries(entries):
    return flavor_class(keys[0][1])
  return "_".join(_block_label(slots) for _, slots, _ in keys)


def operator_info(irrep, multiple, symmetry, spatial_irrep="A1g"):
  """OpPy ``OperatorInfo`` string for a single-site baryon."""
  return "{} {}, {} 0, 0; E, {};{}, 0".format(irrep, multiple, spatial_irrep, symmetry, symmetry)


def _trivial_spatial_irrep(little_group_name):
  return "A1g" if little_group_name == "Oh" else "A1"


def _next_index(irrep_group, prefix):
  taken = [int(name.rsplit("_", 1)[1]) for name in irrep_group
           if name.rsplit("_", 1)[0] == prefix]
  return max(taken) + 1 if taken else 0


def _encode(entries):
  """``(i_vals, d_vals, attrs)`` for one operator."""
  keys0 = next(iter(entries))
  d_vals = []
  for coeff in entries.values():
    d_vals.extend((coeff.real, coeff.imag))

  if len(keys0) == 1:
    slots = keys0[0][1]
    if any(keys[0][1] != slots for keys in entries):
      raise ValueError("All terms of a single-hadron operator must have the same quark content")
    n = len(slots)
    i_vals = [len(entries), 2 * n]
    for keys in entries:
      i_vals.extend(keys[0][2])
      i_vals.extend([0] * n)
    layout = " ".join(["s{}".format(i) for i in range(1, n + 1)]
                      + ["d{}".format(i) for i in range(1, n + 1)])
    attrs = {"slots": [readable_flavor(f) for f in slots]}
    return i_vals, d_vals, attrs, layout

  sizes = tuple(len(slots) for _, slots, _ in keys0)
  if any(tuple(len(slots) for _, slots, _ in keys) != sizes for keys in entries):
    raise NotImplementedError("Terms with differently ordered hadron types cannot share a layout")
  table = sorted({f for keys in entries for _, slots, _ in keys for f in slots},
                 key=lambda f: (_is_anti(f), _flavor_rank(f)))
  code = {f: i for i, f in enumerate(table)}
  i_vals = [len(entries), sum(3 + 3 * n for n in sizes)]
  for keys in entries:
    for momentum, slots, spins in keys:
      i_vals.extend(momentum if momentum is not None else (0, 0, 0))
      i_vals.extend(code[f] for f in slots)
      i_vals.extend(spins)
      i_vals.extend([0] * len(slots))
  block_layouts = []
  for b, n in enumerate(sizes, 1):
    names = ["p{}{}".format(b, x) for x in "xyz"]
    for part in "fsd":
      names += ["{}{}_{}".format(part, b, i) for i in range(1, n + 1)]
    block_layouts.append(" ".join(names))
  attrs = {"flavor_table": [readable_flavor(f) for f in table]}
  return i_vals, d_vals, attrs, " | ".join(block_layouts)


def write_elemental_operator(group, info, entries, attrs=None):
  """Write one operator (``operator_elementals`` output) into ``group``."""
  i_vals, d_vals, layout_attrs, layout = _encode(entries)
  group.create_dataset("OperatorInfo", data=info)
  group.create_dataset("ivals", data=np.array(i_vals, dtype=np.int64))
  group.create_dataset("dvals", data=np.array(d_vals, dtype=np.float64))
  group.attrs["layout"] = layout
  for key, val in dict(layout_attrs, **(attrs or {})).items():
    if isinstance(val, list):
      val = np.array(val, dtype=h5py.string_dtype())
    group.attrs[key] = val


def write_projected_operators_h5(op_rep, irreps, filename, irrep_matrices=None,
                                 flavors=None, root=None, channel=None, mode="a",
                                 convention="oppy", use_generators=False):
  """Project ``op_rep`` onto ``irreps`` and write the operators in OpPy's HDF5 layout.

  Works for every construction in the package: single baryons, mesons, compact
  tetraquarks, dibaryons, tribaryons and meson-meson operators, at rest or in
  flight.

  Parameters
  ----------
  op_rep : OperatorRepresentation
  irreps : iterable[str]
      Little-group irreps to write, e.g. ``("G1g", "G1u", "Hg", "Hu")``.
  filename : str
      Output ``.h5`` file.
  irrep_matrices : callable | None
      Irrep accessor; defaults to ``op_rep.getIrrepAccessor(include_odd_parity=True)``.
  flavors : str | None
      Slot flavors of a single-hadron basis, e.g. ``"uud"``; see ``slot_order``.
  root : str | None
      Top-level group; defaults to ``baryon_operators`` (OpPy), ``meson_operators``,
      ``tetraquark_operators``, ``dibaryon_operators``, ``tribaryon_operators``,
      ``meson_meson_operators`` or ``multi_hadron_operators`` from the operator content.
  channel : str | None
      Second-level group; defaults to the OpPy flavor class for single baryons
      and to the slot flavors (e.g. ``ubarc_dbarc``) otherwise.
  mode : str
      h5py file mode; the default ``"a"`` appends to existing files, numbering
      new operators after those already present.
  convention : str
      ``"oppy"`` (default) reproduces OpPy's operator choice for single baryons
      and its H row phases; ``"native"`` keeps this package's choices.

  Returns
  -------
  list[str]
      HDF5 paths of the written operator groups.
  """
  if irrep_matrices is None:
    irrep_matrices = op_rep.getIrrepAccessor(include_odd_parity=True)

  basis_ops = list(op_rep.basis.operators)
  little_group = op_rep.little_group.little_group
  momentum = op_rep.momentum
  attrs = dict()
  if momentum is not None:
    attrs["momentum"] = np.array([int(p) for p in momentum], dtype=np.int64)

  written = []
  with h5py.File(filename, mode) as h5:
    for irrep in irreps:
      rows = projected_operator_rows(op_rep, irrep, irrep_matrices, use_generators, convention)
      n_mult = len(rows[1])
      multiplets = [[None] * len(rows) for _ in range(n_mult)]
      for row, coeff_rows in rows.items():
        for multiple, coeff_row in enumerate(coeff_rows):
          multiplets[multiple][row - 1] = operator_elementals(
              _combine_coefficients(basis_ops, coeff_row), flavors
          )

      # (group, prefix) -> [next operator index, multiples written in this call]
      counters = dict()
      for multiplet in multiplets:
        multiplet = normalize_multiplet(multiplet)
        first = multiplet[0]
        single = len(next(iter(first))) == 1
        prefix = SINGLE_HADRON_PREFIX if single else MULTI_HADRON_PREFIX
        group_path = "{}/{}".format(root or _default_root(first), channel or _default_channel(first))
        channel_group = h5.require_group(group_path)
        counter = counters.setdefault((group_path, prefix), [
            _next_index(channel_group.require_group("{}_1".format(irrep)), prefix), 0])
        index, multiple = counter
        counter[0] += 1
        counter[1] += 1
        if _is_single_baryon_entries(first):
          info = operator_info(irrep, multiple, flavor_class(next(iter(first))[0][1]),
                               _trivial_spatial_irrep(little_group))
        else:
          info = "{} {}; {}".format(irrep, multiple, channel_group.name.rsplit("/", 1)[1])

        for row, entries in enumerate(multiplet, 1):
          irrep_group = channel_group.require_group("{}_{}".format(irrep, row))
          op_group = irrep_group.create_group("{}_{}".format(prefix, index))
          extra = dict(attrs, little_group=little_group)
          if single:
            extra["flavors"] = _block_label(next(iter(entries))[0][1])
          write_elemental_operator(op_group, info, entries, extra)
          written.append(op_group.name)
  return written


# # # # Reading # # # #

def read_operator_group(group):
  """``{integers of one term: coeff}`` for an operator group (as OpPy's ``op_gr2dict``)."""
  ivals = group["ivals"][()]
  dvals = group["dvals"][()]
  n_terms, n_ints = int(ivals[0]), int(ivals[1])
  return {
      tuple(int(v) for v in ivals[2 + i * n_ints: 2 + (i + 1) * n_ints]):
          dvals[2 * i] + 1j * dvals[2 * i + 1]
      for i in range(n_terms)
  }


def _attr_strings(value):
  return [v.decode() if isinstance(v, bytes) else str(v) for v in value]


def read_operator(group):
  """Decode an operator group into ``[(coeff, [hadron, ...]), ...]``.

  Each hadron is a dict with ``momentum``, ``flavors``, ``spins`` and
  ``displacements``.  Single-hadron groups take the momentum and flavors from
  the group attributes (``None`` if absent, as in files written by OpPy).
  """
  terms = []
  if "flavor_table" in group.attrs:
    table = _attr_strings(group.attrs["flavor_table"])
    sizes = [len(block.split()) // 3 - 1 for block in str(group.attrs["layout"]).split("|")]
    for ints, coeff in read_operator_group(group).items():
      hadrons, pos = [], 0
      for n in sizes:
        block = ints[pos: pos + 3 + 3 * n]
        hadrons.append({
            "momentum": tuple(block[:3]),
            "flavors": tuple(table[f] for f in block[3:3 + n]),
            "spins": tuple(block[3 + n:3 + 2 * n]),
            "displacements": tuple(block[3 + 2 * n:]),
        })
        pos += 3 + 3 * n
      terms.append((coeff, hadrons))
    return terms

  momentum = tuple(int(p) for p in group.attrs["momentum"]) if "momentum" in group.attrs else None
  slots = tuple(_attr_strings(group.attrs["slots"])) if "slots" in group.attrs else None
  for ints, coeff in read_operator_group(group).items():
    n = len(ints) // 2
    terms.append((coeff, [{"momentum": momentum, "flavors": slots,
                           "spins": ints[:n], "displacements": ints[n:]}]))
  return terms
