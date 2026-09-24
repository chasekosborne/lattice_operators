# dibaryon operator construction pipeline

from collections import OrderedDict
from math import isqrt

from sympy import Array, Eijk, Matrix, S, simplify

from .operators import ColorIdx, DiracIdx, Operator, OperatorRepresentation
from .cubic_rotations import LittleGroup, Momentum, P0, spinor_representation
from .tensors import Gamma, GammaRep

_GAMMA = Gamma()
_PARITY_PLUS = _GAMMA.parityPlus
_C = _GAMMA.chargeConj

DIBARYON_SPIN_DIRAC_STRUCTURES = (
    Array(_C * _GAMMA.five * _PARITY_PLUS),
    Array(_C * _GAMMA.one * _PARITY_PLUS),
    Array(_C * _GAMMA.two * _PARITY_PLUS),
    Array(_C * _GAMMA.three * _PARITY_PLUS),
)

SPIN_SINGLET = (0,)
SPIN_TRIPLET = (1, 2, 3)
SPIN_ALL = (0, 1, 2, 3)


def baryon_field(q1, q2, q3, alpha):
    a = ColorIdx("color_a_baryon")
    b = ColorIdx("color_b_baryon")
    c = ColorIdx("color_c_baryon")
    iB = DiracIdx("dirac_i_baryon")
    jB = DiracIdx("dirac_j_baryon")
    return (
        Eijk(a, b, c)
        * q2[a, iB]
        * DIBARYON_SPIN_DIRAC_STRUCTURES[0][iB, jB]
        * q3[b, jB]
        * q1[c, alpha]
    )


def two_baryon_spin_components(baryon_a_alpha, baryon_b_alpha, alpha):
    components = []
    for spin_idx in range(4):
        Cmat = DIBARYON_SPIN_DIRAC_STRUCTURES[spin_idx]
        op_sum = S.Zero
        for i_int in range(4):
            for j_int in range(4):
                coeff = Cmat[i_int, j_int]
                if coeff == 0:
                    continue
                a_op = Operator(baryon_a_alpha.subs(alpha, i_int))
                b_op = Operator(baryon_b_alpha.subs(alpha, j_int))
                op_sum = op_sum + coeff * (a_op * b_op)
        components.append(op_sum)
    return components


# Clebsch-Gordan helpers


def _coeff(channel, weight, spin_components):
    if simplify(weight) == 0:
        return None
    coeff = S(weight)
    return [coeff * comp for comp in spin_components]


def _add_term_lists(term_lists):
    if not term_lists:
        raise ValueError("at least one term list is required")
    out = list(term_lists[0])
    for terms in term_lists[1:]:
        if len(terms) != len(out):
            raise ValueError("isospin term lists must have the same length")
        out = [a + b for a, b in zip(out, terms)]
    return out


def isospin_combinations(channel_components, channel_isospin, target_total_isospin):
    target_I, target_I3 = target_total_isospin

    candidates = OrderedDict()
    for label, components in channel_components.items():
        if label not in channel_isospin:
            raise KeyError("missing isospin assignment for channel {}".format(label))
        I_a, I3_a, I_b, I3_b = channel_isospin[label]
        if I3_a + I3_b != target_I3:
            continue
        candidates[label] = components

    if not candidates:
        raise ValueError(
            "no flavor channels match target total I_3 = {}".format(target_I3)
        )

    if len(candidates) == 1:
        ((label, comps),) = candidates.items()
        I_a, I3_a, I_b, I3_b = channel_isospin[label]
        if target_I < abs(I_a - I_b) or target_I > I_a + I_b:
            raise ValueError(
                "target I={} not allowed for I_a={} I_b={}".format(
                    target_I, I_a, I_b
                )
            )
        return list(comps)

    if len(candidates) == 2 and target_I3 == 0:
        labels = list(candidates.keys())
        if target_I == 0:
            return _add_term_lists([
                candidates[labels[0]],
                [-c for c in candidates[labels[1]]],
            ])
        if target_I == 1:
            return _add_term_lists([
                candidates[labels[0]],
                candidates[labels[1]],
            ])

    raise NotImplementedError(
        "Generic Clebsch-Gordan reduction for {} channels and "
        "(I, I_3) = ({}, {}) is not yet implemented; "
        "build the linear combination explicitly using "
        "_add_term_lists / _coeff helpers.".format(
            len(candidates), target_I, target_I3
        )
    )

_MOMENTUM_SHELL_CACHE = {}
_SPIN_HALF_IRREP_CACHE = {}

def momentum_shell(n):
    if n < 0:
        return []
    if n in _MOMENTUM_SHELL_CACHE:
        return list(_MOMENTUM_SHELL_CACHE[n])

    shell = []
    if n == 0:
        shell.append(Momentum([0, 0, 0]))
    else:
        bound = isqrt(n) + 1
        for x in range(-bound, bound + 1):
            for y in range(-bound, bound + 1):
                for z in range(-bound, bound + 1):
                    if x * x + y * y + z * z == n:
                        shell.append(Momentum([x, y, z]))

    _MOMENTUM_SHELL_CACHE[n] = list(shell)
    return list(shell)


def momentum_shell_pairs(n_a, n_b, total_momentum):
    pairs = []
    for p_a in momentum_shell(n_a):
        p_b = total_momentum - p_a
        if p_b.psq == n_b:
            pairs.append((p_a, p_b))
    return pairs


def _format_momentum(momentum):
    return "({},{},{})".format(momentum.x, momentum.y, momentum.z)


def _format_irrep_content(contents):
    pieces = []
    for irrep, mult in contents:
        if mult == 1:
            pieces.append(irrep)
        else:
            pieces.append("{}*{}".format(mult, irrep))
    return "+".join(pieces) if pieces else "unknown"


def _spin_half_irrep_content(momentum):
    key = tuple(momentum.reduced_pref)
    if key in _SPIN_HALF_IRREP_CACHE:
        return _SPIN_HALF_IRREP_CACHE[key]

    little_group = LittleGroup(False, momentum)
    old_gamma_rep = spinor_representation.gammaRep
    spinor_representation.gammaRep = GammaRep.DIRAC_PAULI
    try:
        # Positive-parity (upper) 2-spinor block of the Dirac–Pauli representation.
        rep = {}
        for rotation in little_group.elements:
            spinor_mat = Matrix(spinor_representation.rotation(rotation, False))
            rep[rotation] = Matrix(spinor_mat[:2, :2])

        contents = []
        for irrep in little_group.irreps:
            mult = S.Zero
            for rotation in little_group.elements:
                mult += (
                    little_group.getCharacter(irrep, rotation).conjugate()
                    * rep[rotation].trace()
                )
            mult = simplify(mult / S(little_group.order))
            if mult == 0:
                continue
            mult_int = int(mult)
            if mult != mult_int:
                raise ValueError(
                    "spin-1/2 irrep multiplicity is not integral for momentum {}".format(
                        momentum
                    )
                )
            contents.append((irrep, mult_int))
    finally:
        spinor_representation.gammaRep = old_gamma_rep

    info = {
        "little_group": str(little_group),
        "irreps": tuple(contents),
    }
    _SPIN_HALF_IRREP_CACHE[key] = info
    return info


def single_baryon_dirac_rows(baryon_alpha, alpha, momentum, n_rows=2):
    """Return the leading ``n_rows`` Dirac components as operators at ``momentum``.

    For spin-1/2 interpolators in the Dirac–Pauli representation the upper
    components (0,1) span the positive-parity little-group irrep (``G1g`` at
    rest, ``G1``/``G`` on moving rays).  Using these components — rather than
    independently LG-projecting at every ray — keeps an Oh orbit of products
    closed under cubic rotations, which is required for dibaryon projection.
    """
    return [
        Operator(baryon_alpha.subs(alpha, i), momentum) for i in range(n_rows)
    ]


def single_baryon_irrep_ops(baryon_alpha, alpha, momentum, irrep, include_odd_parity=True):
    """Project a free-Dirac baryon field onto one little-group irrep at ``momentum``.

    Returns a list ``ops[row-1]`` of independent operators for each irrep row.
    At rest this recovers Dirac components 0,1 as ``G1g`` rows.  At nonzero
    momentum the rows may be linear combinations (``OperatorAdd``) of Dirac
    components; those combinations are oriented with reference-ray irrep
    matrices.
    """
    from .cubic_rotations import E as _E

    dirac_ops = [Operator(baryon_alpha.subs(alpha, i), momentum) for i in range(4)]
    rep = OperatorRepresentation(*dirac_ops)
    accessor = rep.getDiracPauliIrrepAccessor(include_odd_parity=include_odd_parity)
    dim = int(rep.little_group.getCharacter(irrep, _E))

    row_ops = []
    for row in range(1, dim + 1):
        projected = rep.getLinearlyIndependentProjectedOperators(
            irrep, row=row, irrep_matrices=accessor, use_generators=True
        )
        if not projected:
            raise ValueError(
                "No independent operators for {} row {} at momentum {}".format(
                    irrep, row, momentum
                )
            )
        # Take the positive-parity / first copy when multiple G1 copies appear
        # (upper vs lower Dirac).
        row_ops.append(projected[0])
    return row_ops


def irrep_product_basis(
    baryon_a_alpha,
    baryon_b_alpha,
    alpha,
    total_momentum,
    momentum_shells,
    irrep_a,
    irrep_b,
    identical_prune=True,
    use_dirac_rows=True,
):
    """Build a dibaryon basis as products of single-baryon LG-irrep operators.

    This matches the factorization used by external BB operator tables
    (``mom1_Λ1_mom2_Λ2``): each basis element is
    ``B_{Λ_a,row_a}(p_a) * B_{Λ_b,row_b}(p_b)`` with ``p_a + p_b = P_tot``.

    By default ``use_dirac_rows=True`` takes the upper Dirac components as the
    spin-1/2 irrep rows.  Set it to ``False`` to LG-project at each momentum
    separately (correct for a single ray; the resulting products need not be
    closed under the full octahedral group when several rays are combined).
    """
    basis = []
    labels = {}
    for n_a, n_b in momentum_shells:
        pairs = momentum_shell_pairs(n_a, n_b, total_momentum)
        for p_a, p_b in pairs:
            if use_dirac_rows:
                ops_a = single_baryon_dirac_rows(baryon_a_alpha, alpha, p_a)
                ops_b = single_baryon_dirac_rows(baryon_b_alpha, alpha, p_b)
            else:
                ops_a = single_baryon_irrep_ops(baryon_a_alpha, alpha, p_a, irrep_a)
                ops_b = single_baryon_irrep_ops(baryon_b_alpha, alpha, p_b, irrep_b)
            for r_a, op_a in enumerate(ops_a, start=1):
                for r_b, op_b in enumerate(ops_b, start=1):
                    prod = op_a * op_b
                    label = "{ia}_r{ra}{pa}__{ib}_r{rb}{pb}".format(
                        ia=irrep_a,
                        ra=r_a,
                        pa=_format_momentum(p_a),
                        ib=irrep_b,
                        rb=r_b,
                        pb=_format_momentum(p_b),
                    )
                    basis.append(prod)
                    labels[repr(prod)] = label

    if not basis:
        raise ValueError("Empty irrep-product basis for shells {}".format(momentum_shells))

    if identical_prune:
        basis, labels = Dibaryon._prune_to_independent_basis(basis, labels)

    return basis, labels


class IrrepProductDibaryon:
    """Dibaryon operators built from single-baryon LG-irrep products.

    Prefer this construction when comparing against external CG / projection
    coefficient tables that factor operators as
    ``(p1, Λ1, row1) ⊗ (p2, Λ2, row2)``.
    """

    def __init__(
        self,
        baryon_a_alpha,
        baryon_b_alpha,
        alpha,
        total_momentum,
        momentum_shells,
        irrep_a,
        irrep_b,
        channel_label="BB",
        identical_prune=True,
    ):
        self._total_momentum = total_momentum
        self._irrep_a = irrep_a
        self._irrep_b = irrep_b
        self._channel_label = channel_label
        self._basis, self._labels = irrep_product_basis(
            baryon_a_alpha,
            baryon_b_alpha,
            alpha,
            total_momentum,
            momentum_shells,
            irrep_a,
            irrep_b,
            identical_prune=identical_prune,
        )
        # Prefix channel label onto stored labels for printing.
        self._labels = {
            k: "{}_{}".format(channel_label, v) for k, v in self._labels.items()
        }
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

    def little_group_contents(self, nice=True, use_generators=False):
        return self._representation.littleGroupContents(
            nice=nice, use_generators=use_generators
        )

    def get_irrep_accessor(self):
        return self._representation.getBosonicIrrepAccessor()

    def get_projection_matrix(self, irrep, row=1, irrep_matrices=None, use_generators=False):
        return self._representation.getProjectionMatrix(
            irrep,
            row=row,
            irrep_matrices=irrep_matrices,
            use_generators=use_generators,
        )

    def get_projected_operators(
        self, irrep, row=1, irrep_matrices=None, use_generators=False
    ):
        return self._representation.getLinearlyIndependentProjectedOperators(
            irrep,
            row=row,
            irrep_matrices=irrep_matrices,
            use_generators=use_generators,
        )

    def print_projected_operators(self, irreps, irrep_matrices, use_generators=False):
        self._representation.print_projected_operators_raw(
            irreps,
            irrep_matrices,
            operator_labels=self._labels,
            use_generators=use_generators,
        )

    def projected_coefficient_rows(
        self, irrep, row=1, irrep_matrices=None, use_generators=False
    ):
        return self._representation.getLinearlyIndependentProjectedCoefficientRows(
            irrep,
            row=row,
            irrep_matrices=irrep_matrices,
            use_generators=use_generators,
        )


def _constituent_metadata(label, momentum, shell):
    info = _spin_half_irrep_content(momentum)
    return "{label}{mom}[n={shell}; {lg}; {irrep}]".format(
        label=label,
        mom=_format_momentum(momentum),
        shell=shell,
        lg=info["little_group"],
        irrep=_format_irrep_content(info["irreps"]),
    )


def dibaryon_basis(spin_components, total_momentum, momentum_shells, spin_indices):
    basis = []
    for n_a, n_b in momentum_shells:
        pairs = momentum_shell_pairs(n_a, n_b, total_momentum)
        if not pairs:
            continue
        for p_a, p_b in pairs:
            for spin_idx in spin_indices:
                comp = spin_components[spin_idx]
                basis.append(comp.projectMomentum(p_a, p_b))
    return basis


class Dibaryon:
    def __init__(
        self,
        spin_components,
        total_momentum,
        momentum_shells,
        spin_indices=SPIN_SINGLET,
        channel_label="dibaryon",
        prune_redundant=True,
    ):
        self._spin_components = list(spin_components)
        self._total_momentum = total_momentum
        self._momentum_shells = tuple((int(a), int(b)) for a, b in momentum_shells)
        self._spin_indices = tuple(int(s) for s in spin_indices)
        self._channel_label = channel_label

        raw_basis = dibaryon_basis(
            self._spin_components,
            self._total_momentum,
            self._momentum_shells,
            self._spin_indices,
        )
        if not raw_basis:
            raise ValueError(
                "Empty dibaryon basis: no momentum pair satisfies the requested "
                "shells {} for P_tot = {}".format(
                    list(self._momentum_shells), total_momentum
                )
            )

        raw_labels = self._make_basis_labels(raw_basis)

        if prune_redundant:
            self._basis, self._labels = self._prune_to_independent_basis(
                raw_basis, raw_labels
            )
        else:
            self._basis = raw_basis
            self._labels = raw_labels

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
        return self._total_momentum

    def _make_basis_labels(self, basis):
        labels = {}
        spin_tags = {0: "S0", 1: "Sx", 2: "Sy", 3: "Sz"}
        slot = 0
        for n_a, n_b in self._momentum_shells:
            pairs = momentum_shell_pairs(n_a, n_b, self._total_momentum)
            for p_a, p_b in pairs:
                a_meta = _constituent_metadata("pa", p_a, n_a)
                b_meta = _constituent_metadata("pb", p_b, n_b)
                for spin_idx in self._spin_indices:
                    op = basis[slot]
                    label = "{label}_{spin}_{a}_{b}".format(
                        label=self._channel_label,
                        spin=spin_tags[spin_idx],
                        a=a_meta,
                        b=b_meta,
                    )
                    labels[repr(op)] = label
                    slot += 1
        return labels

    @staticmethod
    def _prune_to_independent_basis(raw_basis, raw_labels):
        from operators.operators import OperatorBasis

        if not raw_basis:
            return list(raw_basis), dict(raw_labels)

        full_basis = OperatorBasis(*raw_basis)
        kept = []
        kept_indices = []
        running = None
        for idx, op in enumerate(raw_basis):
            vec = Matrix(full_basis.vector(op))
            trial = vec if running is None else running.row_join(vec)
            if int(trial.rank()) > (int(running.rank()) if running is not None else 0):
                kept.append(op)
                kept_indices.append(idx)
                running = trial

        kept_label_map = {}
        keys = list(raw_labels.keys())
        for keep_idx in kept_indices:
            key = repr(raw_basis[keep_idx])
            if key in raw_labels:
                kept_label_map[key] = raw_labels[key]
            else:
                kept_label_map[key] = keys[keep_idx] if keep_idx < len(keys) else key

        return kept, kept_label_map

    def little_group_contents(self, nice=True, use_generators=False):
        return self._representation.littleGroupContents(
            nice=nice, use_generators=use_generators
        )

    def get_irrep_accessor(self):
        return self._representation.getBosonicIrrepAccessor()

    def get_projection_matrix(self, irrep, row=1, irrep_matrices=None, use_generators=False):
        return self._representation.getProjectionMatrix(
            irrep,
            row=row,
            irrep_matrices=irrep_matrices,
            use_generators=use_generators,
        )

    def get_projected_operators(
        self, irrep, row=1, irrep_matrices=None, use_generators=False
    ):
        return self._representation.getLinearlyIndependentProjectedOperators(
            irrep,
            row=row,
            irrep_matrices=irrep_matrices,
            use_generators=use_generators,
        )

    def print_projected_operators(self, irreps, irrep_matrices, use_generators=False):
        self._representation.print_projected_operators_raw(
            irreps,
            irrep_matrices,
            operator_labels=self._labels,
            use_generators=use_generators,
        )


def lambda_like_spinor(u, d, s):
    alpha = DiracIdx("alpha_lambda")
    return baryon_field(s, u, d, alpha), alpha


def nucleon_p_spinor(u, d):
    alpha = DiracIdx("alpha_proton")
    return baryon_field(u, u, d, alpha), alpha


def nucleon_n_spinor(u, d):
    alpha = DiracIdx("alpha_neutron")
    return baryon_field(d, u, d, alpha), alpha


def cascade_0_spinor(u, s):
    alpha = DiracIdx("alpha_xi0")
    return baryon_field(u, s, s, alpha), alpha


def cascade_minus_spinor(d, s):
    alpha = DiracIdx("alpha_xim")
    return baryon_field(d, s, s, alpha), alpha


__all__ = [
    "DIBARYON_SPIN_DIRAC_STRUCTURES",
    "SPIN_SINGLET",
    "SPIN_TRIPLET",
    "SPIN_ALL",
    "Dibaryon",
    "IrrepProductDibaryon",
    "baryon_field",
    "two_baryon_spin_components",
    "isospin_combinations",
    "momentum_shell",
    "momentum_shell_pairs",
    "dibaryon_basis",
    "single_baryon_irrep_ops",
    "single_baryon_dirac_rows",
    "irrep_product_basis",
    "lambda_like_spinor",
    "nucleon_p_spinor",
    "nucleon_n_spinor",
    "cascade_0_spinor",
    "cascade_minus_spinor",
]
