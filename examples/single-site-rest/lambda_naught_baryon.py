from sympy import *
from context import operators
from operators.operators import *
from operators.cubic_rotations import *

u = QuarkField.create('u')
d = QuarkField.create('d')
s = QuarkField.create('s')

a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')
i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')

# Table 4.8 single-site Lambda^0: alpha < beta
lambda_naught = Eijk(a, b, c) * (u[a, i] * d[b, j] * s[c, k] - d[a, i] * u[b, j] * s[c, k])

lambda_basis = []
for alpha in range(4):
  for beta in range(alpha + 1, 4):
    for gamma in range(4):
      lambda_basis.append((alpha, beta, gamma))

ops = list()
op_labels = dict()
for alpha, beta, gamma in lambda_basis:
  op = lambda_naught.subs({i: alpha, j: beta, k: gamma})
  op_obj = Operator(op, Momentum([0, 0, 0]))
  ops.append(op_obj)
  op_labels[repr(op_obj)] = f"Lambda_{{{alpha+1}{beta+1}{gamma+1}}}"

op_rep = OperatorRepresentation(*ops)
irrep_accessor = op_rep.getDiracPauliIrrepAccessor(include_odd_parity=True)

op_rep.print_projected_operators_raw(
    ("G1g", "G1u", "Hg", "Hu"),
    irrep_accessor,
    operator_labels=op_labels,
)
