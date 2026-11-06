from sympy import *
from operators.operators import *
from operators.cubic_rotations import *
from operators.tetraquark import (
    ANTIDIQUARK_SCALAR,
    DIQUARK_AXIAL,
    MESON_PSEUDOSCALAR,
    MESON_VECTOR,
    DiquarkAntidiquark,
    MesonMeson,
    TetraquarkOperators,
)

c = QuarkField.create("c")
ubar = AntiQuarkField.create("u")
dbar = AntiQuarkField.create("d")

P001 = Momentum([0, 0, 1])

compact = DiquarkAntidiquark(
    c, c, ubar, dbar,
    total_momentum=P001,
    diquark_gammas=DIQUARK_AXIAL,
    antidiquark_gammas=ANTIDIQUARK_SCALAR,
    channel_label="Tcc",
)
print("Little group:", compact.little_group.little_group)
print("Compact content:", compact.little_group_contents(nice=True, use_generators=True))

DDstar = MesonMeson(
    channels=[(1, (ubar, c), (dbar, c)), (-1, (dbar, c), (ubar, c))],
    gammas_a=MESON_PSEUDOSCALAR,
    gammas_b=MESON_VECTOR,
    total_momentum=P001,
    momentum_shells=[(0, 1), (1, 0)],
    channel_label="DDs",
)
print("D D* content:", DDstar.little_group_contents(nice=True, use_generators=True))

combined = TetraquarkOperators(
    compact.basis + DDstar.basis,
    {**compact.basis_labels, **DDstar.basis_labels},
)
print("Combined content:", combined.little_group_contents(nice=True, use_generators=True))
acc = combined.get_irrep_accessor()
combined.print_projected_operators(("A2", "E"), acc, use_generators=True)

compact.write_h5(("A2", "E"), "tcc_p001_operators.h5", mode="w", use_generators=True)
DDstar.write_h5(("A2", "E"), "tcc_p001_operators.h5", use_generators=True)
