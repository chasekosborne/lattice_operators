from pprint import pprint
from sympy import symbols

from operators.operators import QuarkField
from operators.cubic_rotations import P0
from operators.tribaryon import (
    TRIBARYON_SINGLET_AB,
    TriBaryon,
    nucleon_p_spinor,
    nucleon_n_spinor,
    three_baryon_spin_components,
)

u = QuarkField.create("u")
d = QuarkField.create("d")

p1_field, alpha_p1 = nucleon_p_spinor(u, d, label="p1")   # first proton (uud)
p2_field, alpha_p2 = nucleon_p_spinor(u, d, label="p2")   # second proton (uud)
n_field,  alpha_n  = nucleon_n_spinor(u, d, label="n")    # neutron (dud)

from sympy import symbols as _sym

p2_unified = p2_field.subs(alpha_p2, alpha_p1)
n_unified  = n_field.subs(alpha_n,  alpha_p1)

spin_components = three_baryon_spin_components(
    p1_field, p2_unified, n_unified, alpha_p1
)

print("Number of non-zero spin components: {}".format(len(spin_components)))

tri_rest = TriBaryon(
    spin_components=spin_components,
    total_momentum=P0,
    momentum_shells=[(0, 0, 0)],
    spin_indices=TRIBARYON_SINGLET_AB,
    channel_label="ppn",
)

acc = tri_rest.get_irrep_accessor()

print("\n--- ppn rest (0,0,0) shells, AB-singlet spin ---")
print("Little-group contents:")
print(tri_rest.little_group_contents(nice=True, use_generators=True))

fermionic_irreps_pos = ("G1g", "G2g", "Hg")
fermionic_irreps_neg = ("G1u", "G2u", "Hu")

print("\nProjected operators (positive parity):")
tri_rest.print_projected_operators(fermionic_irreps_pos, acc, use_generators=True)

print("\nProjected operators (negative parity):")
tri_rest.print_projected_operators(fermionic_irreps_neg, acc, use_generators=True)

print("\n--- ppn rest (0,0,0)+(1,1,2)+(1,2,1)+(2,1,1) shells, AB-singlet spin ---")
tri_rest_multi = TriBaryon(
    spin_components=spin_components,
    total_momentum=P0,
    momentum_shells=[(0, 0, 0), (1, 1, 2), (1, 2, 1), (2, 1, 1)],
    spin_indices=TRIBARYON_SINGLET_AB,
    channel_label="ppn",
)
acc_multi = tri_rest_multi.get_irrep_accessor()
print("Little-group contents:")
print(tri_rest_multi.little_group_contents(nice=True, use_generators=True))
