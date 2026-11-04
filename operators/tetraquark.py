# tetraquark (two-quark, two-antiquark) operator construction pipeline

from sympy import Eijk, S, eye

from .operators import Operator, OperatorRepresentation
from .cubic_rotations import P0
from .dibaryon import Dibaryon, _format_momentum, _write_h5, momentum_shell_pairs
from .tensors import Gamma

_GAMMA = Gamma()
_C = _GAMMA.chargeConj
_GAMMA_I = (_GAMMA.one, _GAMMA.two, _GAMMA.three)

# Dirac matrices Gamma for meson bilinears qbar Gamma q, keyed by name.
MESON_GAMMAS = {
    "1": eye(4),
    "g4": _GAMMA.four,
    "g5": _GAMMA.five,
    "g4g5": _GAMMA.four * _GAMMA.five,
    "g1": _GAMMA.one,
    "g2": _GAMMA.two,
    "g3": _GAMMA.three,
    "g4g1": _GAMMA.four * _GAMMA.one,
    "g4g2": _GAMMA.four * _GAMMA.two,
    "g4g3": _GAMMA.four * _GAMMA.three,
    "g5g1": _GAMMA.five * _GAMMA.one,
    "g5g2": _GAMMA.five * _GAMMA.two,
    "g5g3": _GAMMA.five * _GAMMA.three,
}

MESON_SCALAR = ("1",)
MESON_PSEUDOSCALAR = ("g5",)
MESON_VECTOR = ("g1", "g2", "g3")
MESON_AXIAL_VECTOR = ("g5g1", "g5g2", "g5g3")

# Diquark structures C Gamma for eps_abc q_b^T (C Gamma) q_c, which is a color 3bar.
DIQUARK_GAMMAS = {
    "Cg5": _C * _GAMMA.five,
    "Cg1": _C * _GAMMA.one,
    "Cg2": _C * _GAMMA.two,
    "Cg3": _C * _GAMMA.three,
}

# Antidiquark structures Gamma C for eps_ade qbar_d (Gamma C) qbar_e^T, a color 3.
ANTIDIQUARK_GAMMAS = {
    "g5C": _GAMMA.five * _C,
    "g1C": _GAMMA.one * _C,
    "g2C": _GAMMA.two * _C,
    "g3C": _GAMMA.three * _C,
}

DIQUARK_SCALAR = ("Cg5",)
DIQUARK_AXIAL = ("Cg1", "Cg2", "Cg3")
ANTIDIQUARK_SCALAR = ("g5C",)
ANTIDIQUARK_AXIAL = ("g1C", "g2C", "g3C")


def _nonzero_entries(matrix):
  return [(a, b, matrix[a, b]) for a in range(4) for b in range(4) if matrix[a, b] != 0]


def _structure(table, gamma):
  if isinstance(gamma, str):
    try:
      return table[gamma]
    except KeyError:
      raise KeyError("unknown Dirac structure '{}'; choose from {}".format(gamma, list(table)))
  return gamma


def meson_field(qbar, q, gamma):
  """Color-singlet bilinear ``sum_c qbar[c, a] Gamma_ab q[c, b]``.

  ``qbar`` is an ``AntiQuarkField``, ``q`` a ``QuarkField`` and ``gamma`` a
  name from ``MESON_GAMMAS`` or an explicit 4x4 matrix.
  """
  entries = _nonzero_entries(_structure(MESON_GAMMAS, gamma))
  expr = S.Zero
  for c in range(3):
    for a, b, coeff in entries:
      expr += coeff * qbar[c, a] * q[c, b]
  return expr


def diquark_antidiquark_field(q1, q2, qbar3, qbar4, diquark_gamma, antidiquark_gamma):
  """Compact tetraquark ``D_a(q1 q2) Dbar_a(qbar3 qbar4)`` at a single site.

  ``D_a = eps_abc q1[b, i] (C Gamma)_ij q2[c, j]`` and
  ``Dbar_a = eps_ade qbar3[d, k] (Gamma' C)_kl qbar4[e, l]``, so the color sum
  over ``a`` gives ``eps_abc eps_ade = delta_bd delta_ce - delta_be delta_cd``.
  Structures are names from ``DIQUARK_GAMMAS`` / ``ANTIDIQUARK_GAMMAS`` or
  explicit 4x4 matrices.
  """
  dq = _nonzero_entries(_structure(DIQUARK_GAMMAS, diquark_gamma))
  adq = _nonzero_entries(_structure(ANTIDIQUARK_GAMMAS, antidiquark_gamma))
  expr = S.Zero
  for a in range(3):
    for b in range(3):
      for c in range(3):
        eps_abc = Eijk(a, b, c)
        if eps_abc == 0:
          continue
        for d in range(3):
          for e in range(3):
            color = eps_abc * Eijk(a, d, e)
            if color == 0:
              continue
            for i, j, c_dq in dq:
              for k, l, c_adq in adq:
                expr += (color * c_dq * c_adq
                         * q1[b, i] * q2[c, j] * qbar3[d, k] * qbar4[e, l])
  return expr


def diquark_antidiquark_basis(q1, q2, qbar3, qbar4, total_momentum=P0,
                              diquark_gammas=DIQUARK_AXIAL,
                              antidiquark_gammas=ANTIDIQUARK_SCALAR,
                              channel_label="DDbar"):
  """All compact operators for the chosen diquark x antidiquark structures.

  Returns ``(operators, labels)`` with ``labels[repr(op)]`` a printable name.
  """
  ops, labels = [], {}
  for dq in diquark_gammas:
    for adq in antidiquark_gammas:
      op = Operator(diquark_antidiquark_field(q1, q2, qbar3, qbar4, dq, adq), total_momentum)
      ops.append(op)
      labels[repr(op)] = "{}_[{}]x[{}]{}".format(
          channel_label, dq, adq, _format_momentum(total_momentum))
  return ops, labels


def meson_meson_basis(channels, gammas_a, gammas_b, total_momentum, momentum_shells,
                      channel_label="MM"):
  """Two-meson operators ``M_a(p_a) M_b(p_b)`` with ``p_a + p_b = P_tot``.

  Parameters
  ----------
  channels : list[tuple]
      Flavor channels ``(weight, (qbar_a, q_a), (qbar_b, q_b))``; each basis
      element is ``sum weight * M_a M_b`` over the channels, e.g. an isospin
      combination such as ``D^0 D*^+ - D^+ D*^0``.
  gammas_a, gammas_b : sequence
      Dirac structures (names from ``MESON_GAMMAS`` or matrices) of each meson.
  momentum_shells : list[tuple[int, int]]
      ``(n_a, n_b)`` with ``p_a^2 = n_a`` and ``p_b^2 = n_b``.

  Returns ``(operators, labels)``.
  """
  ops, labels = [], {}
  for n_a, n_b in momentum_shells:
    for p_a, p_b in momentum_shell_pairs(n_a, n_b, total_momentum):
      for g_a in gammas_a:
        for g_b in gammas_b:
          op = S.Zero
          for weight, (qbar_a, q_a), (qbar_b, q_b) in channels:
            m_a = Operator(meson_field(qbar_a, q_a, g_a), p_a)
            m_b = Operator(meson_field(qbar_b, q_b, g_b), p_b)
            op = op + weight * (m_a * m_b)
          if op == S.Zero:
            continue
          ops.append(op)
          labels[repr(op)] = "{}_[{}]{}x[{}]{}".format(
              channel_label, g_a, _format_momentum(p_a), g_b, _format_momentum(p_b))
  return ops, labels


class TetraquarkOperators:
  """Projection container for any basis of tetraquark operators at one ``P_tot``.

  Accepts compact (diquark-antidiquark) operators, meson-meson products, or
  both together.  Vanishing and linearly dependent operators are pruned by
  default so the representation matrices stay full rank.
  """

  def __init__(self, operators, labels=None, prune_redundant=True, channel_label=None):
    self._channel_label = channel_label
    operators = list(operators)
    labels = dict(labels or {})
    if not operators:
      raise ValueError("Empty tetraquark basis")
    if prune_redundant:
      operators, labels = Dibaryon._prune_to_independent_basis(operators, labels)
    if not operators:
      raise ValueError("All tetraquark operators vanish")
    self._basis = operators
    self._labels = labels
    self._representation = OperatorRepresentation(*self._basis)

  @property
  def basis(self):
    return list(self._basis)

  @property
  def basis_labels(self):
    return dict(self._labels)

  @property
  def representation(self):
    return self._representation

  @property
  def little_group(self):
    return self._representation.little_group

  @property
  def total_momentum(self):
    return self._representation.momentum

  def little_group_contents(self, nice=True, use_generators=False):
    return self._representation.littleGroupContents(nice=nice, use_generators=use_generators)

  def get_irrep_accessor(self):
    return self._representation.getBosonicIrrepAccessor()

  def get_projection_matrix(self, irrep, row=1, irrep_matrices=None, use_generators=False):
    return self._representation.getProjectionMatrix(
        irrep, row=row, irrep_matrices=irrep_matrices, use_generators=use_generators)

  def get_projected_operators(self, irrep, row=1, irrep_matrices=None, use_generators=False):
    return self._representation.getLinearlyIndependentProjectedOperators(
        irrep, row=row, irrep_matrices=irrep_matrices, use_generators=use_generators)

  def projected_coefficient_rows(self, irrep, row=1, irrep_matrices=None, use_generators=False):
    return self._representation.getLinearlyIndependentProjectedCoefficientRows(
        irrep, row=row, irrep_matrices=irrep_matrices, use_generators=use_generators)

  def print_projected_operators(self, irreps, irrep_matrices, use_generators=False):
    self._representation.print_projected_operators_raw(
        irreps, irrep_matrices, operator_labels=self._labels, use_generators=use_generators)

  def write_h5(self, irreps, filename, irrep_matrices=None, mode="a", use_generators=False,
               **kwargs):
    """Write the projected operators to HDF5; see ``write_projected_operators_h5``.

    Compact operators go under ``tetraquark_operators`` and meson-meson operators
    under ``meson_meson_operators``; ``channel_label`` names the group below.
    """
    return _write_h5(self._representation, irreps, filename, irrep_matrices,
                     self._channel_label, mode, use_generators, **kwargs)


class DiquarkAntidiquark(TetraquarkOperators):
  """Compact ``[q1 q2][qbar3 qbar4]`` tetraquark operators; see ``diquark_antidiquark_basis``."""

  def __init__(self, q1, q2, qbar3, qbar4, total_momentum=P0,
               diquark_gammas=DIQUARK_AXIAL, antidiquark_gammas=ANTIDIQUARK_SCALAR,
               channel_label="DDbar", prune_redundant=True):
    ops, labels = diquark_antidiquark_basis(
        q1, q2, qbar3, qbar4, total_momentum, diquark_gammas, antidiquark_gammas,
        channel_label)
    super().__init__(ops, labels, prune_redundant, channel_label)


class MesonMeson(TetraquarkOperators):
  """Two-meson tetraquark operators over momentum shells; see ``meson_meson_basis``."""

  def __init__(self, channels, gammas_a, gammas_b, total_momentum, momentum_shells,
               channel_label="MM", prune_redundant=True):
    ops, labels = meson_meson_basis(
        channels, gammas_a, gammas_b, total_momentum, momentum_shells, channel_label)
    if not ops:
      raise ValueError(
          "Empty meson-meson basis: no momentum pair satisfies the requested shells {} "
          "for P_tot = {}".format(list(momentum_shells), total_momentum))
    super().__init__(ops, labels, prune_redundant, channel_label)


__all__ = [
    "MESON_GAMMAS",
    "MESON_SCALAR",
    "MESON_PSEUDOSCALAR",
    "MESON_VECTOR",
    "MESON_AXIAL_VECTOR",
    "DIQUARK_GAMMAS",
    "ANTIDIQUARK_GAMMAS",
    "DIQUARK_SCALAR",
    "DIQUARK_AXIAL",
    "ANTIDIQUARK_SCALAR",
    "ANTIDIQUARK_AXIAL",
    "meson_field",
    "diquark_antidiquark_field",
    "diquark_antidiquark_basis",
    "meson_meson_basis",
    "TetraquarkOperators",
    "DiquarkAntidiquark",
    "MesonMeson",
]
