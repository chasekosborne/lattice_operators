from sympy import *
from operators.operators import *
from operators.cubic_rotations import *
from operators.dibaryon import (
    SPIN_SINGLET,
    Dibaryon,
    baryon_field,
    isospin_combinations,
    two_baryon_spin_components,
)

u = QuarkField.create("u")
d = QuarkField.create("d")
s = QuarkField.create("s")
alpha = DiracIdx("alpha_lambda")
lam_a = baryon_field(s, u, d, alpha)
lam_b = baryon_field(s, u, d, alpha)
LL = two_baryon_spin_components(lam_a, lam_b, alpha)
H = isospin_combinations(
    channel_components={"LL": LL},
    channel_isospin={"LL": (0, 0, 0, 0)},
    target_total_isospin=(0, 0),
)

P001 = P([0, 0, 1])
dib001 = Dibaryon(
    spin_components=H,
    total_momentum=P001,
    momentum_shells=[(0, 1), (1, 2), (2, 3)],
    spin_indices=SPIN_SINGLET,
    channel_label="H_LL",
)
acc001 = dib001.representation.getBosonicIrrepAccessor()
P001_A1 = dib001.get_projection_matrix("A1", row=1, irrep_matrices=acc001, use_generators=True)
print("P001 A1 row 1:")
pprint(P001_A1)
dib001.print_projected_operators(("A1", "A2", "E"), acc001, use_generators=True)

P011 = P([0, 1, 1])
dib011 = Dibaryon(
    spin_components=H,
    total_momentum=P011,
    momentum_shells=[(1, 1), (2, 2), (1, 3)],
    spin_indices=SPIN_SINGLET,
    channel_label="H_LL",
)
acc011 = dib011.representation.getBosonicIrrepAccessor()
P011_A1 = dib011.get_projection_matrix("A1", row=1, irrep_matrices=acc011, use_generators=True)
print("P011 A1 row 1:")
pprint(P011_A1)
dib011.print_projected_operators(("A1", "A2", "B1", "B2"), acc011, use_generators=True)

P111 = P([1, 1, 1])
dib111 = Dibaryon(
    spin_components=H,
    total_momentum=P111,
    momentum_shells=[(1, 2), (2, 1), (2, 3)],
    spin_indices=SPIN_SINGLET,
    channel_label="H_LL",
)
acc111 = dib111.representation.getBosonicIrrepAccessor()
P111_A1 = dib111.get_projection_matrix("A1", row=1, irrep_matrices=acc111, use_generators=True)
print("P111 A1 row 1:")
pprint(P111_A1)
dib111.print_projected_operators(("A1", "A2", "E"), acc111, use_generators=True)
