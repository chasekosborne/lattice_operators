import pytest
from sympy import S

from context import operators  # noqa: F401
from operators.cubic_rotations import Momentum, P0
from operators.operators import AntiQuarkField, Operator, OperatorRepresentation, QuarkField
from operators.tetraquark import (
    ANTIDIQUARK_AXIAL,
    ANTIDIQUARK_SCALAR,
    DIQUARK_AXIAL,
    DIQUARK_SCALAR,
    MESON_AXIAL_VECTOR,
    MESON_PSEUDOSCALAR,
    MESON_SCALAR,
    MESON_VECTOR,
    DiquarkAntidiquark,
    MesonMeson,
    meson_field,
)

u, d, c = QuarkField.create("u"), QuarkField.create("d"), QuarkField.create("c")
ubar, dbar = AntiQuarkField.create("u"), AntiQuarkField.create("d")

DDSTAR_I0 = [(1, (ubar, c), (dbar, c)), (-1, (dbar, c), (ubar, c))]


@pytest.mark.parametrize("gammas, expected", [
    (MESON_SCALAR, "A1g"),
    (MESON_PSEUDOSCALAR, "A1u"),
    (MESON_VECTOR, "T1u"),
    (MESON_AXIAL_VECTOR, "T1g"),
])
def test_meson_bilinears_at_rest(gammas, expected):
  ops = [Operator(meson_field(ubar, d, g), P0) for g in gammas]
  assert OperatorRepresentation(*ops).littleGroupContents(nice=True) == expected


def test_identical_pions_back_to_back_are_even_waves_only():
  pipi = MesonMeson([(1, (dbar, u), (dbar, u))], MESON_PSEUDOSCALAR, MESON_PSEUDOSCALAR,
                    P0, [(1, 1)])
  assert len(pipi.basis) == 3
  assert pipi.little_group_contents() == "A1g + Eg"


def test_identical_pions_at_rest():
  pipi = MesonMeson([(1, (dbar, u), (dbar, u))], MESON_PSEUDOSCALAR, MESON_PSEUDOSCALAR,
                    P0, [(0, 0)])
  assert pipi.little_group_contents() == "A1g"


def test_meson_products_commute():
  pi_p = Operator(meson_field(dbar, u, "g5"), Momentum([0, 0, 1]))
  pi_m = Operator(meson_field(dbar, u, "g5"), Momentum([0, 0, -1]))
  ab, ba = (pi_p * pi_m).coefficients, (pi_m * pi_p).coefficients
  assert set(ab) == set(ba)
  assert all(ab[k] - ba[k] == 0 for k in ab)


def test_scalar_diquark_of_identical_quarks_vanishes():
  with pytest.raises(ValueError):
    DiquarkAntidiquark(c, c, ubar, dbar, P0, DIQUARK_SCALAR, ANTIDIQUARK_SCALAR)


def test_tcc_compact_quantum_numbers():
  axial_scalar = DiquarkAntidiquark(c, c, ubar, dbar, P0, DIQUARK_AXIAL, ANTIDIQUARK_SCALAR)
  assert axial_scalar.little_group_contents() == "T1g"
  axial_axial = DiquarkAntidiquark(c, c, ubar, dbar, P0, DIQUARK_AXIAL, ANTIDIQUARK_AXIAL)
  assert axial_axial.little_group_contents() == "A1g + Eg + T1g + T2g"


def test_tcc_compact_moving_subduces_t1g():
  moving = DiquarkAntidiquark(c, c, ubar, dbar, Momentum([0, 0, 1]),
                              DIQUARK_AXIAL, ANTIDIQUARK_SCALAR)
  assert moving.little_group.little_group == "C4v"
  assert moving.little_group_contents(use_generators=True) == "A2 + E"


def test_ddstar_at_rest_is_s_wave_t1g():
  ddstar = MesonMeson(DDSTAR_I0, MESON_PSEUDOSCALAR, MESON_VECTOR, P0, [(0, 0)])
  assert len(ddstar.basis) == 3
  assert ddstar.little_group_contents() == "T1g"
  rows = ddstar.projected_coefficient_rows("T1g", row=2,
                                           irrep_matrices=ddstar.get_irrep_accessor())
  assert len(rows) == 1
  assert sum(1 for x in rows[0] if x != S.Zero) == 1
