"""Two-baryon (dibaryon) operator construction pipeline."""

from collections import OrderedDict
from math import isqrt

from sympy import Array, Eijk, Matrix, S, simplify

from .operators import ColorIdx, DiracIdx, Operator, OperatorRepresentation
from .cubic_rotations import Momentum
from .tensors import Gamma

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


# --- Isospin Clebsch-Gordan helpers -----------------------------------------


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
                pa_str = "({},{},{})".format(p_a.x, p_a.y, p_a.z)
                pb_str = "({},{},{})".format(p_b.x, p_b.y, p_b.z)
                for spin_idx in self._spin_indices:
                    op = basis[slot]
                    label = "{label}_{spin}_pa{pa}_pb{pb}".format(
                        label=self._channel_label,
                        spin=spin_tags[spin_idx],
                        pa=pa_str,
                        pb=pb_str,
                    )
                    labels[repr(op)] = label
                    slot += 1
        return labels

    @staticmethod
    def _prune_to_independent_basis(raw_basis, raw_labels):
        """Drop linearly dependent basis vectors so that the OperatorRepresentation
        carries a full-rank rotation matrix.

        Identical-particle channels (e.g. ``Lambda Lambda``) generate
        ``OperatorMul`` pairs ``(p_a, p_b)`` and ``(p_b, p_a)`` that are equal
        up to a sign from the Grassmann anticommutation of two odd
        building blocks; the raw basis is then rank-deficient.  The
        OperatorRepresentation rotation matrix construction relies on
        invertibility, so we keep only a maximally linearly independent
        subset of the raw basis vectors.
        """
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
        """Return the bosonic irrep-matrix accessor for this dibaryon."""
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
    """Return a single-baryon spinor of Lambda flavor structure (sud)."""
    alpha = DiracIdx("alpha_lambda")
    return baryon_field(s, u, d, alpha), alpha


def nucleon_p_spinor(u, d):
    """Return a proton-like single-baryon spinor (uud)."""
    alpha = DiracIdx("alpha_proton")
    return baryon_field(u, u, d, alpha), alpha


def nucleon_n_spinor(u, d):
    """Return a neutron-like single-baryon spinor (dud)."""
    alpha = DiracIdx("alpha_neutron")
    return baryon_field(d, u, d, alpha), alpha


def cascade_0_spinor(u, s):
    """Return a Xi^0-like single-baryon spinor (uss)."""
    alpha = DiracIdx("alpha_xi0")
    return baryon_field(u, s, s, alpha), alpha


def cascade_minus_spinor(d, s):
    """Return a Xi^--like single-baryon spinor (dss)."""
    alpha = DiracIdx("alpha_xim")
    return baryon_field(d, s, s, alpha), alpha


__all__ = [
    "DIBARYON_SPIN_DIRAC_STRUCTURES",
    "SPIN_SINGLET",
    "SPIN_TRIPLET",
    "SPIN_ALL",
    "Dibaryon",
    "baryon_field",
    "two_baryon_spin_components",
    "isospin_combinations",
    "momentum_shell",
    "momentum_shell_pairs",
    "dibaryon_basis",
    "lambda_like_spinor",
    "nucleon_p_spinor",
    "nucleon_n_spinor",
    "cascade_0_spinor",
    "cascade_minus_spinor",
]
