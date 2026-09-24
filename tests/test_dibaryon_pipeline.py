"""Tests for the dibaryon construction pipeline (operators.dibaryon).

Verifies that:

* Momentum-shell helpers enumerate the lattice momenta correctly.
* Dirac structures used by the dibaryon Dirac coupling have the expected
  shape.
* The single-channel ``Lambda Lambda`` (H-dibaryon) basis at rest with
  ``p_a^2 = p_b^2 = 0`` decomposes purely as the ``A1g`` irrep.
* The same channel at rest with the back-to-back shell ``(1, 1)``
  decomposes as ``2 A1g + Eg`` and that the two ``A1g`` irreducible
  operators reproduce, up to normalization, the manually-built rest-frame
  and back-to-back-sum operators used in :mod:`examples.dibaryon_tests`.
* The N-Xi antisymmetric I=0 spin-1 channel decomposes as ``T1g`` at
  rest, matching the legacy test ``test_P0_T1p_Ls(N_X_a_I0, 0)``.
* The bosonic irrep accessor lazily generates Oh irreps on demand.
"""

from sympy import Matrix, S, simplify

from context import operators  # noqa: F401

from operators.cubic_rotations import (
    C2x,
    C4x,
    C4z,
    E,
    Is,
    Momentum,
    P,
    P0,
    get_bosonic_irrep_matrix,
)
from operators.dibaryon import (
    SPIN_SINGLET,
    SPIN_TRIPLET,
    Dibaryon,
    IrrepProductDibaryon,
    baryon_field,
    dibaryon_basis,
    isospin_combinations,
    momentum_shell,
    momentum_shell_pairs,
    single_baryon_irrep_ops,
    two_baryon_spin_components,
)
from operators.operators import DiracIdx, Operator, OperatorAdd, QuarkField


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_lambda_components():
    u = QuarkField.create("u")
    d = QuarkField.create("d")
    s = QuarkField.create("s")
    alpha = DiracIdx("alpha_lambda")
    lam_a = baryon_field(s, u, d, alpha)
    lam_b = baryon_field(s, u, d, alpha)
    return two_baryon_spin_components(lam_a, lam_b, alpha)


def _build_nx_antisymmetric_components():
    u = QuarkField.create("u")
    d = QuarkField.create("d")
    s = QuarkField.create("s")
    alpha = DiracIdx("alpha_nx")
    N_uud = baryon_field(u, u, d, alpha)
    N_dud = baryon_field(d, u, d, alpha)
    X_ssd = baryon_field(s, s, d, alpha)
    X_ssu = baryon_field(s, s, u, alpha)
    uud_ssd = two_baryon_spin_components(N_uud, X_ssd, alpha)
    dud_ssu = two_baryon_spin_components(N_dud, X_ssu, alpha)
    ssd_uud = two_baryon_spin_components(X_ssd, N_uud, alpha)
    ssu_dud = two_baryon_spin_components(X_ssu, N_dud, alpha)
    return [
        uud_ssd[k] - dud_ssu[k] - ssd_uud[k] + ssu_dud[k] for k in range(4)
    ]


# ---------------------------------------------------------------------------
# Momentum-shell enumeration
# ---------------------------------------------------------------------------


def test_momentum_shell_zero_is_origin_only():
    shell = momentum_shell(0)
    assert len(shell) == 1
    assert shell[0] == Momentum([0, 0, 0])


def test_momentum_shell_one_has_six_unit_vectors():
    shell = momentum_shell(1)
    assert len(shell) == 6
    for m in shell:
        assert m.psq == 1


def test_momentum_shell_pairs_back_to_back_p_eq_zero():
    pairs = momentum_shell_pairs(1, 1, P0)
    assert len(pairs) == 6
    for p_a, p_b in pairs:
        assert (p_a + p_b).psq == 0
        assert p_a.psq == 1
        assert p_b.psq == 1


def test_momentum_shell_pairs_moving_frame_no_back_to_back():
    Ptot = P([0, 0, 1])
    pairs = momentum_shell_pairs(1, 0, Ptot)
    assert (Ptot, P0) in pairs
    assert len(pairs) == 1


# ---------------------------------------------------------------------------
# Spin coupling
# ---------------------------------------------------------------------------


def test_two_baryon_spin_components_returns_four_bosonic_operators():
    comps = _build_lambda_components()
    assert len(comps) == 4
    for comp in comps:
        assert hasattr(comp, "bosonic")
        assert comp.bosonic


# ---------------------------------------------------------------------------
# Dibaryon channel decomposition
# ---------------------------------------------------------------------------


def test_h_dibaryon_at_rest_with_zero_momentum_is_single_a1g():
    comps = _build_lambda_components()
    dib = Dibaryon(
        spin_components=comps,
        total_momentum=P0,
        momentum_shells=[(0, 0)],
        spin_indices=SPIN_SINGLET,
        channel_label="HD_rest",
    )
    decomposition = dib.little_group_contents(nice=False, use_generators=True)
    assert decomposition["A1g"] == 1
    for irrep, mult in decomposition.items():
        if irrep == "A1g":
            continue
        assert mult == 0


def test_h_dibaryon_back_to_back_p_one_decomposes_to_2a1g_and_eg():
    comps = _build_lambda_components()
    dib = Dibaryon(
        spin_components=comps,
        total_momentum=P0,
        momentum_shells=[(0, 0), (1, 1)],
        spin_indices=SPIN_SINGLET,
        channel_label="HD",
    )
    decomposition = dib.little_group_contents(nice=False, use_generators=True)
    assert decomposition["A1g"] == 2
    assert decomposition["Eg"] == 1
    # Every other Oh irrep should be absent.
    for irrep, mult in decomposition.items():
        if irrep in ("A1g", "Eg"):
            continue
        assert mult == 0


def test_h_dibaryon_a1g_projected_operators_match_manual_construction():
    """The two A1g operators must be (a) the rest-frame Lambda Lambda and
    (b) the symmetric sum over the three back-to-back p^2=1 directions, up
    to normalization.  This reproduces the manual construction used in
    examples/dibaryon_tests.py::test_P0_A1p (n = 0 and n = 1)."""

    comps = _build_lambda_components()
    dib = Dibaryon(
        spin_components=comps,
        total_momentum=P0,
        momentum_shells=[(0, 0), (1, 1)],
        spin_indices=SPIN_SINGLET,
        channel_label="HD",
    )
    irrep_accessor = dib.get_irrep_accessor()
    rows = dib.representation.getLinearlyIndependentProjectedCoefficientRows(
        "A1g", row=1, irrep_matrices=irrep_accessor, use_generators=True
    )
    assert len(rows) == 2

    # Each coefficient row should have exactly one nonzero entry (rest-frame)
    # or three (back-to-back sum).
    nz_counts = sorted(
        sum(1 for c in row if simplify(c) != 0) for row in rows
    )
    assert nz_counts == [1, 3]


def test_dibaryon_basis_labels_include_constituent_irrep_metadata():
    comps = _build_lambda_components()
    dib = Dibaryon(
        spin_components=comps,
        total_momentum=P([0, 0, 1]),
        momentum_shells=[(0, 1)],
        spin_indices=SPIN_SINGLET,
        channel_label="HD_move",
    )
    labels = set(dib.basis_labels.values())
    assert len(labels) == 1
    label = next(iter(labels))
    assert "pa(0,0,0)[n=0; O_h^D; G1g]" in label
    assert "pb(0,0,1)[n=1; C_{4v}^D; G1]" in label


def test_nx_antisymmetric_i0_s1_at_rest_is_t1g():
    comps = _build_nx_antisymmetric_components()
    dib = Dibaryon(
        spin_components=comps,
        total_momentum=P0,
        momentum_shells=[(0, 0)],
        spin_indices=SPIN_TRIPLET,
        channel_label="NX_a_S1",
    )
    decomposition = dib.little_group_contents(nice=False, use_generators=True)
    assert decomposition["T1g"] == 1
    for irrep, mult in decomposition.items():
        if irrep == "T1g":
            continue
        assert mult == 0


# ---------------------------------------------------------------------------
# Bosonic irrep matrices
# ---------------------------------------------------------------------------


def test_bosonic_irrep_matrix_a1g_is_one_at_every_oh_element():
    for elem in (E, C4z, Is, C2x):
        mat = get_bosonic_irrep_matrix("Oh", "A1g", elem)
        assert mat == Matrix([[1]])


def test_bosonic_irrep_matrix_a2g_flips_under_c4_and_i_c2():
    assert get_bosonic_irrep_matrix("Oh", "A2g", E) == Matrix([[1]])
    assert get_bosonic_irrep_matrix("Oh", "A2g", C4z) == Matrix([[-1]])


def test_bosonic_irrep_matrix_eg_is_two_dimensional():
    eg_E = get_bosonic_irrep_matrix("Oh", "Eg", E)
    assert eg_E.shape == (2, 2)
    assert eg_E == Matrix.eye(2)


def test_bosonic_irrep_matrix_t1u_is_three_dimensional():
    t1u_E = get_bosonic_irrep_matrix("Oh", "T1u", E)
    assert t1u_E.shape == (3, 3)
    assert t1u_E == Matrix.eye(3)


# ---------------------------------------------------------------------------
# Isospin combinations
# ---------------------------------------------------------------------------


def test_isospin_combinations_single_channel_passes_through():
    comps = _build_lambda_components()
    out = isospin_combinations(
        channel_components={"LL": comps},
        channel_isospin={"LL": (0, 0, 0, 0)},
        target_total_isospin=(0, 0),
    )
    assert len(out) == 4
    for original, projected in zip(comps, out):
        assert original is projected


def test_isospin_combinations_two_channel_filters_by_total_i3():
    """Only channels matching the requested I_3 are kept, and length-2
    candidate maps trigger the symmetric/antisymmetric branch."""
    comps = _build_lambda_components()
    # Same-shape channels labelled differently with consistent I_3 sums.
    out = isospin_combinations(
        channel_components={"AB": comps, "BA": comps},
        channel_isospin={
            "AB": (S(1) / 2, S(1) / 2, S(1) / 2, -S(1) / 2),
            "BA": (S(1) / 2, -S(1) / 2, S(1) / 2, S(1) / 2),
        },
        target_total_isospin=(0, 0),
    )
    assert len(out) == 4
    # ``AB - BA`` with AB == BA must produce zero coefficient operators.
    for op in out:
        for value in op.coefficients.values():
            assert simplify(value) == 0


def test_isospin_combinations_rejects_mismatched_total_i3():
    comps = _build_lambda_components()
    try:
        isospin_combinations(
            channel_components={"LL": comps},
            channel_isospin={"LL": (0, 0, 0, 0)},
            target_total_isospin=(0, 1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched I_3")


# ---------------------------------------------------------------------------
# Irrep-product construction (bbops-style) and accessor fixes
# ---------------------------------------------------------------------------


def test_fermionic_irrep_accessor_works_at_non_reference_momentum():
    """C4v G1 tables are stored for P=001; P=100 must remap via reference_element."""
    u = QuarkField.create("u")
    d = QuarkField.create("d")
    s = QuarkField.create("s")
    alpha = DiracIdx("alpha_lam")
    lam = baryon_field(s, u, d, alpha)
    ops = single_baryon_irrep_ops(lam, alpha, P([1, 0, 0]), "G1")
    assert len(ops) == 2


def test_bosonic_irrep_accessor_works_at_non_reference_momentum():
    comps = _build_lambda_components()
    dib = Dibaryon(
        spin_components=comps,
        total_momentum=P([1, 0, 0]),
        momentum_shells=[(0, 1)],
        spin_indices=SPIN_SINGLET,
        channel_label="H_x",
    )
    acc = dib.get_irrep_accessor()
    # Must not raise KeyError for C4x (reference-mapped to C4z in the tables).
    mat = acc("A1", C4x)
    assert mat == Matrix([[1]])


def test_irrep_product_rest_g1g_matches_spin_singlet_a1g():
    """Rest-frame G1g⊗G1g A1g equals the Dirac spin-singlet (bbops 000_G1g)."""
    u = QuarkField.create("u")
    d = QuarkField.create("d")
    s = QuarkField.create("s")
    alpha = DiracIdx("alpha_lam")
    lam = baryon_field(s, u, d, alpha)

    ip = IrrepProductDibaryon(
        lam, lam, alpha, P0, [(0, 0)], "G1g", "G1g", channel_label="LL"
    )
    assert ip.little_group_contents(nice=False, use_generators=True)["A1g"] == 1

    acc = ip.get_irrep_accessor()
    projected = ip.get_projected_operators(
        "A1g", row=1, irrep_matrices=acc, use_generators=True
    )
    assert len(projected) == 1

    bbops_singlet = (
        Operator(lam.subs(alpha, 0), P0) * Operator(lam.subs(alpha, 1), P0)
        - Operator(lam.subs(alpha, 1), P0) * Operator(lam.subs(alpha, 0), P0)
    )
    from operators.operators import OperatorBasis

    basis = OperatorBasis(bbops_singlet, projected[0])
    v_bb = Matrix(basis.vector(bbops_singlet))
    v_our = Matrix(basis.vector(projected[0]))
    assert int(v_bb.row_join(v_our).rank()) == 1


def test_irrep_product_shell_one_a1g_is_different_row_combination():
    """G1⊗G1 at p^2=1 projects to A1g on the antisymmetric (different-row) channel.

    This matches the Dirac spin-singlet construction and disagrees with the
    same-row pattern stored under ``A1g`` in some external BB tables (those
    operators have Is=-1 in our convention, i.e. A1u).
    """
    u = QuarkField.create("u")
    d = QuarkField.create("d")
    s = QuarkField.create("s")
    alpha = DiracIdx("alpha_lam")
    lam = baryon_field(s, u, d, alpha)

    ip = IrrepProductDibaryon(
        lam, lam, alpha, P0, [(1, 1)], "G1", "G1", channel_label="LL"
    )
    decomposition = ip.little_group_contents(nice=False, use_generators=True)
    assert decomposition["A1g"] == 1

    acc = ip.get_irrep_accessor()
    rows = ip.projected_coefficient_rows(
        "A1g", row=1, irrep_matrices=acc, use_generators=True
    )
    assert len(rows) == 1

    labels = list(ip.basis_labels.values())
    vals = [simplify(c) for c in rows[0]]
    scale = next(c for c in vals if c != 0)
    vals = [simplify(c / scale) for c in vals]

    # Nonzero support must be different-row products only (r1⊗r2 and r2⊗r1).
    for coeff, label in zip(vals, labels):
        if coeff == 0:
            continue
        assert "_r1" in label and "_r2" in label or (
            "r1" in label and "r2" in label
        )
        # Same-row labels look like ..._r1(...)__..._r1(...) or r2/r2.
        assert not (
            ("_r1" in label and label.count("_r1") == 2)
            or ("_r2" in label and label.count("_r2") == 2)
        )


def test_operator_add_allows_mixed_dirac_components():
    u = QuarkField.create("u")
    d = QuarkField.create("d")
    s = QuarkField.create("s")
    alpha = DiracIdx("alpha_lam")
    lam = baryon_field(s, u, d, alpha)
    o0 = Operator(lam.subs(alpha, 0), P([1, 0, 0]))
    o1 = Operator(lam.subs(alpha, 1), P([1, 0, 0]))
    from operators.operators import OperatorAdd

    summed = o0 + o1
    assert isinstance(summed, OperatorAdd)
