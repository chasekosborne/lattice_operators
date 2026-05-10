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

dib0 = Dibaryon(
    spin_components=H,
    total_momentum=P0,
    momentum_shells=[(0, 0)],
    spin_indices=SPIN_SINGLET,
    channel_label="H_LL",
)
acc = dib0.representation.getBosonicIrrepAccessor()
P0_A1g = dib0.get_projection_matrix("A1g", row=1, irrep_matrices=acc, use_generators=True)

print("P0 shells (0,0) only, A1g row 1:")
pprint(P0_A1g)
dib0.print_projected_operators(("A1g",), acc, use_generators=True)

dib = Dibaryon(
    spin_components=H,
    total_momentum=P0,
    momentum_shells=[(0, 0), (1, 1), (2, 2), (3, 3)],
    spin_indices=SPIN_SINGLET,
    channel_label="H_LL",
)
acc = dib.representation.getBosonicIrrepAccessor()
P_A1g = dib.get_projection_matrix("A1g", row=1, irrep_matrices=acc, use_generators=True)

print("P0 shells (0,0)+(1,1)+(2,2)+(3,3), A1g row 1:")
pprint(P_A1g)
print("Projected operators:")
dib.print_projected_operators(("A1g",), acc, use_generators=True)
