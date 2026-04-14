from sympy import *
from context import operators
from operators.operators import *
from operators.cubic_rotations import *

u = QuarkField.create('u')
d = QuarkField.create('d')

a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')
i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')

nucleon = Eijk(a,b,c) * (u[a,i]*u[b,j]*d[c,k] - d[a,i]*u[b,j]*u[c,k])

nucleon_basis = []
for alpha in range(1, 4):
  if alpha == 1:
    for beta in range(2):
      nucleon_basis.append((alpha, beta, 0))
  elif alpha == 2:
    for gamma in range(2):
      for beta in range(3):
        nucleon_basis.append((alpha, beta, gamma))
  else:
    for gamma in range(3):
      for beta in range(3):
        nucleon_basis.append((alpha, beta, gamma))
    for gamma in range(3):
      nucleon_basis.append((alpha, 3, gamma))

ops = list()
op_labels = dict()
for alpha,beta,gamma in nucleon_basis:
  op = nucleon.subs({i:alpha,j:beta,k:gamma})
  op_obj = Operator(op, Momentum([0,0,0]))
  ops.append(op_obj)
  op_labels[repr(op_obj)] = f"N_{{{alpha+1}{beta+1}{gamma+1}}}"

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

print("P_G1g_row1:")
pprint(P_G1g_row1)


inner_product_matrix = op_rep.getModifiedInnerProductMatrix()

print("Projected operators:")
op_rep.print_projected_operators_raw(
    ("G1g", "G1u", "Hg", "Hu"),
    irrep_accessor,
    operator_labels=op_labels,
)
