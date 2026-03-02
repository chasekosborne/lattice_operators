# The $\Delta^{++}$ Baryon

To help get started, let us go over a simple example that constructs a set of Delta-like Baryon operators with momentum $\mathbf{P} = (0, 0, 1)$ and then outputs the Little group irrep decomposition.


## Constructing a list of baryon operators
We start by first importing the `operator` module, `cubic_rotations` module, and the sympy package:

```python
from sympy import *
from operators.operators import *
from operators.cubic_rotations import *
```

Next, we need to construct some quark fields that will be used to construct the operator. Since our Delta operator will be constructed from quark fields of flavor 'u', we construct these as:

```python
u = QuarkField.create('u')
```

Note that the operator knows nothing of quark flavor, and the 'u' passed to `create` is just a name.

Next, we need to create some indices:

```python
a = ColorIdx('a')
b = ColorIdx('b')
c = ColorIdx('c')

i = DiracIdx('i')
j = DiracIdx('j')
k = DiracIdx('k')
```

Then, we will construct a set of Delta operators (one for each spin index) as follows:

```python
Delta = Eijk(a,b,c) * u[a,i] * u[b,j] * u[c,k]
```

Note that the color index comes before the Dirac spin index in a quark field.

Finally, we can begin with

```python
ops = list()
for i_int in range(4):
    for j_int in range(i_int, 4):
        for k_int in range(j_int, 4):
            op = Delta.subs({i: i_int, j: j_int, k: k_int})
            ops.append(Operator(op, Momentum([0, 0, 1])))
```

Then, the list of operators represents a possible operator basis, which we pass into the class `OperatorRepresentation`, which will determine the irrep decomposition:

```python
op_rep = OperatorRepresentation(*ops)

print("\nMoving Delta:")
print(op_rep.littleGroupContents(True, False))
```

The output of this code is:

```
6 G1 + 4 G2
```

which tells us that this basis contains 6 copies of the G₁ irrep and 4 copies of the G₂ irrep.
