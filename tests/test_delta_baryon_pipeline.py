from sympy import Eijk, Matrix, S, cancel, expand, gcd, simplify, sign

from context import operators  # noqa: F401
from operators.cubic_rotations import (
    C4y,
    C4z,
    E,
    Is,
    Momentum,
    conjugate_spin_irrep_accessor,
)
from operators.operators import (
    ColorIdx,
    DiracIdx,
    Operator,
    OperatorRepresentation,
    QuarkField,
)


def _build_delta_operator_representation():
    u = QuarkField.create("u")
    a, b, c = ColorIdx("a"), ColorIdx("b"), ColorIdx("c")
    i, j, k = DiracIdx("i"), DiracIdx("j"), DiracIdx("k")
    delta = Eijk(a, b, c) * u[a, i] * u[b, j] * u[c, k]
    ops = []
    for i_int in range(4):
        for j_int in range(i_int, 4):
            for k_int in range(j_int, 4):
                op = delta.subs({i: i_int, j: j_int, k: k_int})
                ops.append(Operator(op, Momentum([0, 0, 0])))
    return OperatorRepresentation(*ops)


def _basis_triples_0based():
    triples = []
    for i_int in range(4):
        for j_int in range(i_int, 4):
            for k_int in range(j_int, 4):
                triples.append((i_int, j_int, k_int))
    assert len(triples) == 20
    return triples


def _coeff_for_triple(row_vec, triple, triples):
    idx = triples.index(triple)
    return simplify(row_vec[idx])


def _matrix_is_zero(mat):
    return all(simplify(entry) == 0 for entry in mat)


def _matrix_equal(a, b):
    return _matrix_is_zero(a - b)


def _format_row_gcd_primitive(row_entries):
    coeffs = [simplify(expand(c, complex=True)) for c in row_entries]
    nonzero = [c for c in coeffs if simplify(c) != 0]
    if not nonzero:
        return
    common = nonzero[0]
    for c in nonzero[1:]:
        common = gcd(common, c)
    if simplify(common) == 0:
        common = S.One
    norm = [cancel(simplify(c / common)) for c in coeffs]
    first_nz = next((v for v in norm if simplify(v) != 0), None)
    if first_nz is not None and simplify(sign(first_nz)) < 0:
        norm = [-v for v in norm]
    return norm


def _hu_row1_rows_paper_order(op_rep, accessor):
    row1_rows = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "Hu", row=1, irrep_matrices=accessor
    )
    triples = _basis_triples_0based()

    def one_hot_sub(row):
        nz = []
        for c, t in zip(row, triples):
            if simplify(c) == 0:
                continue
            nz.append(
                "{}{}{}".format(t[0] + 1, t[1] + 1, t[2] + 1)
            )
        return nz[0] if len(nz) == 1 else None

    if len(row1_rows) == 2:
        s0 = one_hot_sub(list(row1_rows[0]))
        s1 = one_hot_sub(list(row1_rows[1]))
        if s0 == "113" and s1 == "333":
            return [row1_rows[1], row1_rows[0]]
    return row1_rows


# --- Pipeline steps (aligned with the example script) ---


def test_step_basis_twenty_independent_deltas():
    op_rep = _build_delta_operator_representation()
    assert op_rep.dimension == 20
    assert len(list(op_rep.basis.operators)) == 20
    triples = _basis_triples_0based()
    assert triples[0] == (0, 0, 0)
    assert triples[-1] == (3, 3, 3)


def test_step_little_group_is_fermionic_oh_at_rest():
    op_rep = _build_delta_operator_representation()
    assert op_rep.little_group.little_group == "Oh"
    assert op_rep.little_group.fermionic
    assert "Hg" in op_rep.little_group.irreps
    assert "Hu" in op_rep.little_group.irreps


def test_step_representation_matrices_defined_for_generators():
    op_rep = _build_delta_operator_representation()
    for el in (C4y, C4z, Is):
        W = op_rep.getRepresentationMatrix(el, use_generators=True)
        assert W.shape == (20, 20)
        assert W.is_square


def test_step_dirac_pauli_accessor_odd_parity_irreps():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    assert acc("G1g", E).shape == (2, 2)
    assert acc("G1u", E).shape == (2, 2)
    assert acc("Hg", E).shape == (4, 4)
    assert acc("Hu", E).shape == (4, 4)


def test_step_conjugate_spin_accessor_with_identity_is_noop():
    op_rep = _build_delta_operator_representation()
    base = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    wrapped = conjugate_spin_irrep_accessor(base, Matrix.eye(4))
    for R in list(op_rep.little_group.elements)[:5]:
        assert _matrix_equal(wrapped("Hg", R), base("Hg", R))
        assert _matrix_equal(wrapped("Hu", R), base("Hu", R))


def test_step_g1g_projection_is_idempotent():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    P = op_rep.getProjectionMatrix("G1g", row=1, irrep_matrices=acc)
    assert P.shape == (20, 20)
    assert _matrix_equal(P * P, P)
    assert int(P.rank()) == 1


def test_step_hg_row_projections_idempotent_and_pairwise_orthogonal():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    rows = [
        op_rep.getProjectionMatrix("Hg", row=r, irrep_matrices=acc) for r in range(1, 5)
    ]
    for P in rows:
        assert _matrix_equal(P * P, P)
    for i, Pi in enumerate(rows):
        for j, Pj in enumerate(rows):
            if i != j:
                assert _matrix_is_zero(simplify(Pi * Pj))


def test_step_hg_row_sum_equals_character_projector():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    g = op_rep.little_group.order
    d_lambda = op_rep.little_group.getCharacter("Hg", E)
    P_sum = Matrix.zeros(op_rep.dimension)
    for row in range(1, 5):
        P_sum += op_rep.getProjectionMatrix("Hg", row=row, irrep_matrices=acc)
    P_char = Matrix.zeros(op_rep.dimension)
    for R in op_rep.little_group.elements:
        chi = op_rep.little_group.getCharacter("Hg", R)
        P_char += chi.conjugate() * op_rep.getRepresentationMatrix(R, True).T
    P_char = simplify(P_char * S(d_lambda) / S(g))
    assert _matrix_equal(simplify(P_sum), simplify(P_char))


def test_step_d_g1g_one_linearly_independent_coefficient_row():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    rows = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "G1g", row=1, irrep_matrices=acc
    )
    assert len(rows) == 1
    P = op_rep.getProjectionMatrix("G1g", row=1, irrep_matrices=acc)
    assert len(rows) == int(P.rank())


def test_step_d_hg_row1_two_linearly_independent_coefficient_rows():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    rows = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "Hg", row=1, irrep_matrices=acc
    )
    assert len(rows) == 2
    P = op_rep.getProjectionMatrix("Hg", row=1, irrep_matrices=acc)
    assert len(rows) == int(P.rank())
    stacked = Matrix.vstack(*[Matrix([list(r.row(0))]) for r in rows])
    assert int(stacked.rank()) == 2


def test_step_e_g1g_partner_row2_table_410_ratios():
    """Table 4.10: row 1 ``134 - 233``, row 2 ``144 - 234`` (1-based indices)."""
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    triples = _basis_triples_0based()
    r1 = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "G1g", row=1, irrep_matrices=acc
    )
    assert len(r1) == 1
    v1 = list(r1[0])
    pr2 = op_rep.getPartnerRowCoefficientRows(
        "G1g", source_rows=r1, target_row=2, source_row=1, irrep_matrices=acc
    )
    assert len(pr2) == 1
    v2 = list(pr2[0])

    t134, t233 = (0, 2, 3), (1, 2, 2)
    t144, t234 = (0, 3, 3), (1, 2, 3)
    a134 = _coeff_for_triple(v1, t134, triples)
    b233 = _coeff_for_triple(v1, t233, triples)
    assert simplify(a134 / b233) == -1

    a144 = _coeff_for_triple(v2, t144, triples)
    b234 = _coeff_for_triple(v2, t234, triples)
    assert simplify(a144 / b234) == -1


def test_step_e_hg_set2_row2_two_delta_ratio_not_spoiled_by_int_cast():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    triples = _basis_triples_0based()
    r1 = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "Hg", row=1, irrep_matrices=acc
    )
    pr2 = op_rep.getPartnerRowCoefficientRows(
        "Hg", source_rows=r1, target_row=2, source_row=1, irrep_matrices=acc
    )
    v = list(pr2[1])
    t134, t233 = (0, 2, 3), (1, 2, 2)
    c134 = _coeff_for_triple(v, t134, triples)
    c233 = _coeff_for_triple(v, t233, triples)
    assert simplify(c134 / c233) == 2

    norm = _format_row_gcd_primitive(v)
    i134 = triples.index(t134)
    i233 = triples.index(t233)
    assert simplify(norm[i134]) == 2
    assert simplify(norm[i233]) == 1


def test_step_hu_multiplet_swap_puts_set_one_first():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    ordered = _hu_row1_rows_paper_order(op_rep, acc)
    triples = _basis_triples_0based()
    v0 = list(ordered[0])
    nz = [(triples[k], simplify(v0[k])) for k in range(20) if simplify(v0[k]) != 0]
    assert len(nz) == 1
    assert nz[0][0] == (2, 2, 2)  # Delta_333


def test_step_e_hu_set2_row3_two_one_ratio_after_swap():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    triples = _basis_triples_0based()
    row1 = _hu_row1_rows_paper_order(op_rep, acc)
    pr3 = op_rep.getPartnerRowCoefficientRows(
        "Hu", source_rows=row1, target_row=3, source_row=1, irrep_matrices=acc
    )
    v = list(pr3[1])
    # paper Delta_{223} is sorted indices (2,2,3) → 0-based (1,1,2), not (1,2,2) (= 233).
    t124, t223 = (0, 1, 3), (1, 1, 2)
    assert simplify(_coeff_for_triple(v, t124, triples) / _coeff_for_triple(v, t223, triples)) == 2


def test_step_d_g1u_row1_opposes_114_and_123_as_in_table():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    triples = _basis_triples_0based()
    r1 = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "G1u", row=1, irrep_matrices=acc
    )
    assert len(r1) == 1
    v = list(r1[0])
    t123, t114 = (0, 1, 2), (0, 0, 3)
    c123 = _coeff_for_triple(v, t123, triples)
    c114 = _coeff_for_triple(v, t114, triples)
    assert simplify(c123 / c114) == -1


def test_row_mixing_matrix_shape_and_partner_linearity_smoke():
    op_rep = _build_delta_operator_representation()
    acc = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)
    M = op_rep.getRowMixingProjectionMatrix(
        "Hg", target_row=2, source_row=1, irrep_matrices=acc
    )
    assert M.shape == (20, 20)
    r1 = op_rep.getLinearlyIndependentProjectedCoefficientRows(
        "Hg", row=1, irrep_matrices=acc
    )
    for src in r1:
        direct = Matrix([list(src)]) * M
        via_api = op_rep.getPartnerRowCoefficientRows(
            "Hg",
            source_rows=[src],
            target_row=2,
            source_row=1,
            irrep_matrices=acc,
        )[0]
        assert _matrix_equal(direct, via_api)
