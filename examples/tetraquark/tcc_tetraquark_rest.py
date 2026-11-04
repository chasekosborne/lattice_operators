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

# Compact [cc]_{1+} [ubar dbar]_{0+} operators
compact = DiquarkAntidiquark(
    c, c, ubar, dbar,
    total_momentum=P0,
    diquark_gammas=DIQUARK_AXIAL,
    antidiquark_gammas=ANTIDIQUARK_SCALAR,
    channel_label="Tcc",
)
print("Compact basis size:", len(compact.basis))
print("Compact content:", compact.little_group_contents(nice=True, use_generators=True))
acc = compact.get_irrep_accessor()
compact.print_projected_operators(("T1g",), acc, use_generators=True)

# I = 0 meson-meson D D* operators: D^0 D*^+ - D^+ D*^0
DDstar_I0 = [
    (1, (ubar, c), (dbar, c)),
    (-1, (dbar, c), (ubar, c)),
]
DDstar = MesonMeson(
    channels=DDstar_I0,
    gammas_a=MESON_PSEUDOSCALAR,
    gammas_b=MESON_VECTOR,
    total_momentum=P0,
    momentum_shells=[(0, 0), (1, 1)],
    channel_label="DDs",
)
print("D D* basis size:", len(DDstar.basis))
print("D D* content:", DDstar.little_group_contents(nice=True, use_generators=True))

# Combined compact + meson-meson basis
combined = TetraquarkOperators(
    compact.basis + DDstar.basis,
    {**compact.basis_labels, **DDstar.basis_labels},
)
print("Combined content:", combined.little_group_contents(nice=True, use_generators=True))
acc = combined.get_irrep_accessor()
combined.print_projected_operators(("T1g",), acc, use_generators=True)

# compact operators go to tetraquark_operators/, D D* operators to meson_meson_operators/
compact.write_h5(("T1g",), "tcc_rest_operators.h5", mode="w", use_generators=True)
DDstar.write_h5(("T1g",), "tcc_rest_operators.h5", use_generators=True)
