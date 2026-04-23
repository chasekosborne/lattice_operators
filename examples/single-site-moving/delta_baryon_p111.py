from sympy import *
from operators.operators import *
from operators.cubic_rotations import *

P_ref = Momentum([1, 1, 1])

u = QuarkField.create('u')

a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')
i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')

delta = Eijk(a, b, c) * u[a, i] * u[b, j] * u[c, k]

ops = list()
op_labels = dict()
for i_int in range(4):
  for j_int in range(i_int, 4):
    for k_int in range(j_int, 4):
      op = delta.subs({i: i_int, j: j_int, k: k_int})
      op_obj = Operator(op, P_ref)
      ops.append(op_obj)
      op_labels[repr(op_obj)] = f"Delta_{{{i_int+1}{j_int+1}{k_int+1}}}"

op_rep = OperatorRepresentation(*ops)

USE_GENERATORS = False

irrep_accessor = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)

# C3v^D has two 1-dimensional spinor irreps (F1, F2) and one 2-dimensional
# spinor irrep (G).
P_F1_row1 = op_rep.getProjectionMatrix(
    "F1", row=1, irrep_matrices=irrep_accessor, use_generators=USE_GENERATORS
)
P_F2_row1 = op_rep.getProjectionMatrix(
    "F2", row=1, irrep_matrices=irrep_accessor, use_generators=USE_GENERATORS
)
P_G_row1 = op_rep.getProjectionMatrix(
    "G", row=1, irrep_matrices=irrep_accessor, use_generators=USE_GENERATORS
)

print("\nP_F1_row1:")
pprint(P_F1_row1)

print("\nP_F2_row1:")
pprint(P_F2_row1)

print("\nP_G_row1:")
pprint(P_G_row1)

print("\nProjected operators:")
op_rep.print_projected_operators_raw(
    ("F1", "F2", "G"),
    irrep_accessor,
    operator_labels=op_labels,
    use_generators=USE_GENERATORS,
)
