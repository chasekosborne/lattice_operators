from sympy import *
from context import operators
from operators.operators import *
from operators.cubic_rotations import *

u = QuarkField.create('u')
s = QuarkField.create('s')

a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')
i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')

# Table 4.6 single-site Sigma^+: alpha <= beta
sigma_plus = Eijk(a, b, c) * u[a, i] * u[b, j] * s[c, k]

sigma_basis = []
for alpha in range(4):
  for beta in range(alpha, 4):
    for gamma in range(4):
      sigma_basis.append((alpha, beta, gamma))

ops = list()
op_labels = dict()
for alpha, beta, gamma in sigma_basis:
  op = sigma_plus.subs({i: alpha, j: beta, k: gamma})
  op_obj = Operator(op, Momentum([0, 0, 0]))
  ops.append(op_obj)
  op_labels[repr(op_obj)] = f"Sigma_{{{alpha+1}{beta+1}{gamma+1}}}"

op_rep = OperatorRepresentation(*ops)
c4y_mat = op_rep.getRepresentationMatrix(C4y)
c4z_mat = op_rep.getRepresentationMatrix(C4z)
i2_mat = op_rep.getRepresentationMatrix(Is)

irrep_accessor = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)

P_G1g_row1 = op_rep.getProjectionMatrix("G1g", row=1, irrep_matrices=irrep_accessor)

op_rep.print_projected_operators_raw(
    ("G1g", "G1u", "Hg", "Hu"),
    irrep_accessor,
    operator_labels=op_labels,
)
