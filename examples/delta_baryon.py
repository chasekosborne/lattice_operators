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
for i_int in range(4):
  for j_int in range(i_int,4):
    for k_int in range(j_int,4):
      op = delta.subs({i:i_int,j:j_int,k:k_int})
      ops.append(Operator(op, Momentum([0,0,0])))

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
