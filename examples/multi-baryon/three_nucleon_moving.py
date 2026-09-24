from pprint import pprint

from operators.operators import QuarkField
from operators.cubic_rotations import P
from operators.tribaryon import (
    TRIBARYON_SINGLET_AB,
    TriBaryon,
    nucleon_p_spinor,
    nucleon_n_spinor,
    three_baryon_spin_components,
)

u = QuarkField.create("u")
d = QuarkField.create("d")

p1_field, alpha_p1 = nucleon_p_spinor(u, d, label="p1")
p2_field, alpha_p2 = nucleon_p_spinor(u, d, label="p2")
n_field,  alpha_n  = nucleon_n_spinor(u, d, label="n")

p2_unified = p2_field.subs(alpha_p2, alpha_p1)
n_unified  = n_field.subs(alpha_n,  alpha_p1)

spin_components = three_baryon_spin_components(
    p1_field, p2_unified, n_unified, alpha_p1
)

P001 = P([0, 0, 1])

# momentum shell (n_a, n_b, n_c):
#   (0, 0, 1): two baryons at rest, one carrying all the boost
#   (0, 1, 2): one at rest, others on |p|²=1 and |p|²=2 shells
#   (1, 1, 1): three baryons each with |p|²=1 summing to P001

print("\n=== P_tot = (0,0,1) ===")
tri001 = TriBaryon(
    spin_components=spin_components,
    total_momentum=P001,
    momentum_shells=[(0, 0, 1), (0, 1, 2), (1, 1, 1)],
    spin_indices=TRIBARYON_SINGLET_AB,
    channel_label="ppn",
)
acc001 = tri001.get_irrep_accessor()

print("Little-group contents:")
print(tri001.little_group_contents(nice=True, use_generators=True))

print("\nProjected operators:")
tri001.print_projected_operators(
    tri001.little_group.irreps, acc001, use_generators=True
)


P011 = P([0, 1, 1])

print("\n=== P_tot = (0,1,1) ===")
tri011 = TriBaryon(
    spin_components=spin_components,
    total_momentum=P011,
    momentum_shells=[(0, 1, 1), (1, 1, 2), (1, 2, 3)],
    spin_indices=TRIBARYON_SINGLET_AB,
    channel_label="ppn",
)
acc011 = tri011.get_irrep_accessor()

print("Little-group contents:")
print(tri011.little_group_contents(nice=True, use_generators=True))

print("\nProjected operators:")
tri011.print_projected_operators(
    tri011.little_group.irreps, acc011, use_generators=True
)


P111 = P([1, 1, 1])

print("\n=== P_tot = (1,1,1) ===")
tri111 = TriBaryon(
    spin_components=spin_components,
    total_momentum=P111,
    momentum_shells=[(0, 1, 2), (1, 1, 3), (1, 2, 2)],
    spin_indices=TRIBARYON_SINGLET_AB,
    channel_label="ppn",
)
acc111 = tri111.get_irrep_accessor()

print("Little-group contents:")
print(tri111.little_group_contents(nice=True, use_generators=True))

print("\nProjected operators:")
tri111.print_projected_operators(
    tri111.little_group.irreps, acc111, use_generators=True
)
