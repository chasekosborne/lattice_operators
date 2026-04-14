from sympy import Matrix, simplify, S, Eijk

from context import operators  # noqa: F401
from operators.cubic_rotations import E, Momentum, get_spinor_irrep_matrix
from operators.operators import QuarkField, ColorIdx, DiracIdx, Operator, OperatorRepresentation


def _build_delta_representation():
  u = QuarkField.create("u")

  a = ColorIdx("a")
  b = ColorIdx("b")
  c = ColorIdx("c")
  i = DiracIdx("i")
  j = DiracIdx("j")
  k = DiracIdx("k")

  delta = Eijk(a, b, c) * u[a, i] * u[b, j] * u[c, k]

  ops = []
  for i_int in range(4):
    for j_int in range(i_int, 4):
      for k_int in range(j_int, 4):
        op = delta.subs({i: i_int, j: j_int, k: k_int})
        ops.append(Operator(op, Momentum([0, 0, 0])))

  return OperatorRepresentation(*ops)


def _matrix_is_zero(mat):
  return all(simplify(entry) == 0 for entry in mat)


def _matrix_equal(a, b):
  return _matrix_is_zero(a - b)


def _character_projector(op_rep, irrep):
  g = op_rep.little_group.order
  d_lambda = op_rep.little_group.getCharacter(irrep, E)

  proj = Matrix.zeros(op_rep.dimension)
  for R in op_rep.little_group.elements:
    chi = op_rep.little_group.getCharacter(irrep, R)
    proj += chi.conjugate() * op_rep.getRepresentationMatrix(R, True).T

  return simplify(proj * S(d_lambda) / S(g))


def test_delta_projection_g1g_accessor_matches_lazy_getter():
  op_rep = _build_delta_representation()
  acc = op_rep.getDiracPauliIrrepAccessor()
  for R in op_rep.little_group.elements:
    assert _matrix_equal(acc("G1g", R), get_spinor_irrep_matrix("G1g", R))


def test_delta_projection_g1g_row1_is_projector():
  op_rep = _build_delta_representation()
  P = op_rep.getProjectionMatrix("G1g", row=1, irrep_matrices=op_rep.getDiracPauliIrrepAccessor())

  assert P.shape == (op_rep.dimension, op_rep.dimension)
  assert _matrix_equal(P * P, P)
  assert _matrix_equal(P.H, P)


def test_delta_projection_hg_rows_are_projectors_and_orthogonal():
  op_rep = _build_delta_representation()
  irrep_mats = op_rep.getDiracPauliIrrepMatrices()

  hg_rows = [
      op_rep.getProjectionMatrix("Hg", row=row, irrep_matrices=irrep_mats)
      for row in range(1, 5)
  ]

  for P in hg_rows:
    assert P.shape == (op_rep.dimension, op_rep.dimension)
    assert _matrix_equal(P * P, P)
    assert _matrix_equal(P.H, P)

  for i, P_i in enumerate(hg_rows):
    for j, P_j in enumerate(hg_rows):
      if i == j:
        continue
      assert _matrix_is_zero(simplify(P_i * P_j))


def test_delta_projection_hg_row_sum_matches_character_projector():
  op_rep = _build_delta_representation()
  irrep_mats = op_rep.getDiracPauliIrrepMatrices()

  P_hg_rowsum = Matrix.zeros(op_rep.dimension)
  for row in range(1, 5):
    P_hg_rowsum += op_rep.getProjectionMatrix("Hg", row=row, irrep_matrices=irrep_mats)

  P_hg_char = _character_projector(op_rep, "Hg")
  assert _matrix_equal(simplify(P_hg_rowsum), simplify(P_hg_char))


def test_delta_projection_hg_row1_independent_projected_operators_match_rank():
  op_rep = _build_delta_representation()
  irrep_mats = op_rep.getDiracPauliIrrepMatrices()

  projected_ops = op_rep.getLinearlyIndependentProjectedOperators(
      "Hg", row=1, irrep_matrices=irrep_mats
  )
  P_hg_row1 = op_rep.getProjectionMatrix("Hg", row=1, irrep_matrices=irrep_mats)

  assert len(projected_ops) == int(P_hg_row1.rank())
  assert len(projected_ops) > 0

  # the selected operators should be linearly independent in the original basis.
  vecs = [Matrix(op_rep.basis.vector(op)) for op in projected_ops]
  assert Matrix.hstack(*vecs).rank() == len(projected_ops)


def test_delta_projection_hg_row1_independent_rows_match_rank():
  op_rep = _build_delta_representation()
  irrep_mats = op_rep.getDiracPauliIrrepMatrices()

  rows = op_rep.getLinearlyIndependentProjectedCoefficientRows(
      "Hg", row=1, irrep_matrices=irrep_mats
  )
  P_hg_row1 = op_rep.getProjectionMatrix("Hg", row=1, irrep_matrices=irrep_mats)

  assert len(rows) == int(P_hg_row1.rank())
  assert len(rows) > 0
  assert Matrix.vstack(*rows).rank() == len(rows)

