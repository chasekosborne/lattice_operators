# three-baryon operator construction pipeline

from collections import OrderedDict
from math import isqrt

from sympy import Array, Eijk, Matrix, S, simplify

from .operators import ColorIdx, DiracIdx, Operator, OperatorMul, OperatorRepresentation
from .cubic_rotations import Momentum
from .dibaryon import (
    DIBARYON_SPIN_DIRAC_STRUCTURES,
    SPIN_SINGLET,
    SPIN_TRIPLET,
    SPIN_ALL,
    _constituent_metadata,
    baryon_field,
    momentum_shell,
    momentum_shell_pairs,
)

# Matching the dibaryon naming convention for convenience
# a "spin index" for the tribaryon is a (spin_ab, k_c) pair encoded as 4*spin_ab + k_c,
# giving 16 indices total (0..15)
TRIBARYON_SPIN_ALL = tuple(range(16))
TRIBARYON_SINGLET_AB = tuple(range(4))
TRIBARYON_K0 = tuple(4 * s for s in range(4))


def three_baryon_spin_components(baryon_a, baryon_b, baryon_c, alpha):
    components = []
    for spin_ab in range(4):
        Cab = DIBARYON_SPIN_DIRAC_STRUCTURES[spin_ab]
        for k_int in range(4):
            c_op = Operator(baryon_c.subs(alpha, k_int))
            op_sum = S.Zero
            for i_int in range(4):
                for j_int in range(4):
                    coeff = Cab[i_int, j_int]
                    if coeff == 0:
                        continue
                    a_op = Operator(baryon_a.subs(alpha, i_int))
                    b_op = Operator(baryon_b.subs(alpha, j_int))
                    term = OperatorMul(a_op, b_op, c_op)
                    op_sum = op_sum + coeff * term
            if op_sum != S.Zero:
                components.append(op_sum)
    return components


def momentum_shell_triples(n_a, n_b, n_c, total_momentum):
    # p_a² = n_a,  p_b² = n_b,  p_c² = n_c,  p_a + p_b + p_c = total_momentum.
    triples = []
    for p_a in momentum_shell(n_a):
        for p_b in momentum_shell(n_b):
            p_c = total_momentum - p_a - p_b
            if p_c.psq == n_c:
                triples.append((p_a, p_b, p_c))
    return triples


def tribaryon_basis(spin_components, total_momentum, momentum_shells, spin_indices):
    basis = []
    for n_a, n_b, n_c in momentum_shells:
        triples = momentum_shell_triples(n_a, n_b, n_c, total_momentum)
        if not triples:
            continue
        for p_a, p_b, p_c in triples:
            for spin_idx in spin_indices:
                if spin_idx >= len(spin_components):
                    continue
                comp = spin_components[spin_idx]
                basis.append(comp.projectMomentum(p_a, p_b, p_c))
    return basis

# essentially the dibaryon class adapted for 3 baryons
class TriBaryon:
    def __init__(
        self,
        spin_components,
        total_momentum,
        momentum_shells,
        spin_indices=None,
        channel_label="tribaryon",
        prune_redundant=True,
    ):
        self._spin_components = list(spin_components)
        self._total_momentum = total_momentum
        self._momentum_shells = tuple(
            (int(a), int(b), int(c)) for a, b, c in momentum_shells
        )
        if spin_indices is None:
            self._spin_indices = tuple(range(len(self._spin_components)))
        else:
            self._spin_indices = tuple(int(s) for s in spin_indices)
        self._channel_label = channel_label

        raw_basis = tribaryon_basis(
            self._spin_components,
            self._total_momentum,
            self._momentum_shells,
            self._spin_indices,
        )
        if not raw_basis:
            raise ValueError(
                "Empty tribaryon basis: no momentum triple satisfies the requested "
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
        spin_tags = {0: "S0", 1: "Sx", 2: "Sy", 3: "Sz"}
        labels = {}
        slot = 0
        for n_a, n_b, n_c in self._momentum_shells:
            triples = momentum_shell_triples(n_a, n_b, n_c, self._total_momentum)
            for p_a, p_b, p_c in triples:
                a_meta = _constituent_metadata("pa", p_a, n_a)
                b_meta = _constituent_metadata("pb", p_b, n_b)
                c_meta = _constituent_metadata("pc", p_c, n_c)
                for spin_idx in self._spin_indices:
                    if spin_idx >= len(self._spin_components):
                        continue
                    spin_ab = spin_idx // 4
                    k_c = spin_idx % 4
                    op = basis[slot]
                    label = "{label}_{spin}k{kc}_{a}_{b}_{c}".format(
                        label=self._channel_label,
                        spin=spin_tags[spin_ab],
                        kc=k_c,
                        a=a_meta,
                        b=b_meta,
                        c=c_meta,
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
        for keep_idx in kept_indices:
            key = repr(raw_basis[keep_idx])
            if key in raw_labels:
                kept_label_map[key] = raw_labels[key]
            else:
                kept_label_map[key] = key

        return kept, kept_label_map

    def little_group_contents(self, nice=True, use_generators=False):
        return self._representation.littleGroupContents(
            nice=nice, use_generators=use_generators
        )

    def get_irrep_accessor(self):
        """Return the irrep accessor appropriate for fermionic (3-baryon) operators."""
        return self._representation.getDiracPauliIrrepAccessor()

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


def nucleon_p_spinor(u, d, label="proton"):
    alpha = DiracIdx("alpha_{}".format(label))
    return baryon_field(u, u, d, alpha), alpha


def nucleon_n_spinor(u, d, label="neutron"):
    """Neutron baryon field template (dud) with a unique free Dirac index."""
    alpha = DiracIdx("alpha_{}".format(label))
    return baryon_field(d, u, d, alpha), alpha


def lambda_spinor(u, d, s, label="lambda"):
    """Lambda baryon field template (sud) with a unique free Dirac index."""
    alpha = DiracIdx("alpha_{}".format(label))
    return baryon_field(s, u, d, alpha), alpha


def sigma_plus_spinor(u, s, label="sigma_plus"):
    """Sigma+ baryon field template (uus) with a unique free Dirac index."""
    alpha = DiracIdx("alpha_{}".format(label))
    return baryon_field(u, u, s, alpha), alpha


def cascade_0_spinor(u, s, label="xi0"):
    """Xi0 baryon field template (uss) with a unique free Dirac index."""
    alpha = DiracIdx("alpha_{}".format(label))
    return baryon_field(u, s, s, alpha), alpha


def cascade_minus_spinor(d, s, label="xim"):
    """Xi- baryon field template (dss) with a unique free Dirac index."""
    alpha = DiracIdx("alpha_{}".format(label))
    return baryon_field(d, s, s, alpha), alpha



__all__ = [
    "TRIBARYON_SPIN_ALL",
    "TRIBARYON_SINGLET_AB",
    "TRIBARYON_K0",
    "TriBaryon",
    "three_baryon_spin_components",
    "momentum_shell_triples",
    "tribaryon_basis",
    "nucleon_p_spinor",
    "nucleon_n_spinor",
    "lambda_spinor",
    "sigma_plus_spinor",
    "cascade_0_spinor",
    "cascade_minus_spinor",
]
