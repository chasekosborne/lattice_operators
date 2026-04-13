from sympy import *
from context import operators
from operators.operators import *
from operators.cubic_rotations import *

# Creating quark field
u = QuarkField.create('u')

a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')
i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')

delta = Eijk(a,b,c) * u[a,i] * u[b,j] * u[c,k]

ops = list()
op_labels = dict()
for i_int in range(4):
  for j_int in range(i_int,4):
    for k_int in range(j_int,4):
      op = delta.subs({i:i_int,j:j_int,k:k_int})
      op_obj = Operator(op, Momentum([0,0,0]))
      ops.append(op_obj)
      op_labels[repr(op_obj)] = f"Delta_{{{i_int+1}{j_int+1}{k_int+1}}}"

op_rep = OperatorRepresentation(*ops)
c4y_mat = op_rep.getRepresentationMatrix(C4y)
c4z_mat = op_rep.getRepresentationMatrix(C4z)
i2_mat = op_rep.getRepresentationMatrix(Is)

print("C4y:")
pprint(c4y_mat)

print("C4z:")
pprint(c4z_mat)

print("I2:")
pprint(i2_mat)

irrep_accessor = op_rep.getDiracPauliIrrepAccessor(include_odd_parity = True)

P_G1g_row1 = op_rep.getProjectionMatrix("G1g", row=1, irrep_matrices=irrep_accessor)
P_Hg_row1 = op_rep.getProjectionMatrix("Hg", row=1, irrep_matrices=irrep_accessor)


print("P_G1g_row1:")
pprint(P_G1g_row1)


print("P_Hg_row1:")
pprint(P_Hg_row1)

print("Projected operators (gcd-normalized Delta basis):")
op_rep.print_projected_operators_raw(("G1g", "G1u", "Hg", "Hu"), irrep_accessor, operator_labels=op_labels)
