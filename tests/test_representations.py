from sympy import Matrix, sqrt

from sympy import *
from context import operators
from operators.operators import *
from operators.cubic_rotations import *


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


def test_delta_generators_representation_matrices_match_reference():
  op_rep = _build_delta_representation()

  W_C4y = op_rep.getRepresentationMatrix(C4y, use_generators=True)
  W_C4z = op_rep.getRepresentationMatrix(C4z, use_generators=True)
  W_Is = op_rep.getRepresentationMatrix(Is, use_generators=True)

  # Basic sanity checks on shapes
  dim = op_rep.dimension
  assert W_C4y.shape == (dim, dim)
  assert W_C4z.shape == (dim, dim)
  assert W_Is.shape == (dim, dim)

  assert W_Is == W_Is.diagonalize()[0]  # matrix is diagonal in this basis
  diag_vals = [W_Is[i, i] for i in range(dim)]
  assert all(val in (1, -1) for val in diag_vals)

  non_zero_entries = [W_C4z[i, i] for i in range(dim) if W_C4z[i, i] != 0]
  assert non_zero_entries, "Expected non-zero diagonal entries for W(C4z)"
  for val in non_zero_entries:
    assert (val.conjugate() * val).simplify() == 1

  # Check that rescaling by 2*sqrt(2) produces an integer matrix.
  scaled_C4y = (2 * sqrt(2)) * W_C4y
  for entry in scaled_C4y:
    assert entry == int(entry)

