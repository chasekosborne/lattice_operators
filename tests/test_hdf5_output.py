import h5py
import numpy as np
import pytest
from sympy import Eijk

from context import operators  # noqa: F401
from operators.cubic_rotations import Momentum, P0
from operators.dibaryon import SPIN_SINGLET, Dibaryon, baryon_field, two_baryon_spin_components
from operators.hdf5_output import (
    default_flavor_order,
    elemental_coefficients,
    flavor_class,
    operator_elementals,
    read_operator,
    read_operator_group,
    write_projected_operators_h5,
)
from operators.operators import (
    AntiQuarkField,
    ColorIdx,
    DiracIdx,
    Operator,
    OperatorRepresentation,
    QuarkField,
)
from operators.tetraquark import (
    ANTIDIQUARK_SCALAR,
    DIQUARK_AXIAL,
    MESON_PSEUDOSCALAR,
    MESON_VECTOR,
    DiquarkAntidiquark,
    MesonMeson,
    diquark_antidiquark_field,
    meson_field,
)

a, b = ColorIdx("a"), ColorIdx("b")
i, j, k = DiracIdx("i"), DiracIdx("j"), DiracIdx("k")
u, d, s = QuarkField.create("u"), QuarkField.create("d"), QuarkField.create("s")
c = QuarkField.create("c")
ubar, dbar = AntiQuarkField.create("u"), AntiQuarkField.create("d")
cc = ColorIdx("c")


def _sym(terms):
  """Expand ``{(s1, s2, s3): value}`` over all orderings of the Dirac indices (uuu)."""
  out = dict()
  for key, val in terms.items():
    for perm in {(key[p[0]], key[p[1]], key[p[2]]) for p in
                 [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]}:
      out[perm + (0, 0, 0)] = val
  return out


# Operators written by OpPy's construct_baryon_ops.py for the S3+ (uuu) class.
OPPY_DELTA_REFERENCE = {
    "S3+/G1g_1/SS_0": _sym({(1, 3, 4): 1, (2, 3, 3): -2}),
    "S3+/G1g_2/SS_0": _sym({(1, 4, 4): 2, (2, 3, 4): -1}),
    "S3+/Hg_2/SS_0": _sym({(1, 1, 2): 2 * np.sqrt(3)}),
    "S3+/Hg_2/SS_1": _sym({(1, 3, 4): 2, (2, 3, 3): 2}),
    "S3+/Hu_1/SS_1": _sym({(3, 3, 3): 6}),
}


@pytest.fixture(scope="module")
def delta_h5(tmp_path_factory):
  delta = Eijk(a, b, cc) * u[a, i] * u[b, j] * u[cc, k]
  ops = [Operator(delta.subs({i: x, j: y, k: z}), Momentum([0, 0, 0]))
         for x in range(4) for y in range(x, 4) for z in range(y, 4)]
  path = tmp_path_factory.mktemp("h5") / "delta.h5"
  write_projected_operators_h5(OperatorRepresentation(*ops), ("G1g", "G1u", "Hg", "Hu"),
                               str(path), mode="w")
  with h5py.File(path, "r") as h5:
    yield h5


def test_layout_matches_oppy(delta_h5):
  irrep_groups = set(delta_h5["baryon_operators/S3+"])
  assert irrep_groups == {"G1g_1", "G1g_2", "G1u_1", "G1u_2",
                          "Hg_1", "Hg_2", "Hg_3", "Hg_4", "Hu_1", "Hu_2", "Hu_3", "Hu_4"}
  assert set(delta_h5["baryon_operators/S3+/Hg_3"]) == {"SS_0", "SS_1"}

  op = delta_h5["baryon_operators/S3+/Hg_1/SS_1"]
  assert set(op) == {"OperatorInfo", "ivals", "dvals"}
  assert op["OperatorInfo"][()] == b"Hg 1, A1g 0, 0; E, S3+;S3+, 0"
  ivals = op["ivals"][()]
  assert ivals[1] == 6
  assert len(ivals) == 2 + 6 * ivals[0]
  assert len(op["dvals"][()]) == 2 * ivals[0]


@pytest.mark.parametrize("path", sorted(OPPY_DELTA_REFERENCE))
def test_delta_matches_oppy_reference(delta_h5, path):
  mine = read_operator_group(delta_h5["baryon_operators/" + path])
  ref = OPPY_DELTA_REFERENCE[path]
  assert set(mine) == set(ref)
  for key, val in ref.items():
    assert abs(mine[key] - val) < 1e-8


def test_flavor_classes():
  assert default_flavor_order("udu") == ("u", "u", "d")
  assert default_flavor_order("sdu") == ("u", "d", "s")
  assert [flavor_class(f) for f in ("uuu", "uud", "udu", "duu", "uds")] == \
      ["S3+", "Sigma1+", "Sigma2+", "Sigma3+", "E"]


def test_elementals_of_nucleon_like_operator():
  # eps_abc (u_a u_b d_c - d_a u_b u_c) = 2 eps_abc u_a u_b d_c for Dirac (1, 1, 2)
  op = Operator((Eijk(a, b, cc) * (u[a, i] * u[b, j] * d[cc, k] - d[a, i] * u[b, j] * u[cc, k]))
                .subs({i: 0, j: 0, k: 1}), Momentum([0, 0, 0]))
  flavors, elementals = elemental_coefficients(op.coefficients)
  assert flavors == ("u", "u", "d")
  assert set(elementals) == {(1, 1, 2), (2, 1, 1), (1, 2, 1)}
  # coefficients are summed over the u <-> u slot swap, as in OpPy
  assert abs(elementals[(1, 1, 2)] - 2) < 1e-12


def test_rejects_non_singlet_operator():
  op = Operator(u[0, 0] * u[1, 1] * d[1, 2], Momentum([0, 0, 0]))
  with pytest.raises(ValueError):
    elemental_coefficients(op.coefficients)


def _operator_groups(path):
  groups = dict()
  with h5py.File(path, "r") as h5:
    h5.visititems(lambda name, obj: groups.__setitem__(name, read_operator(obj))
                  if isinstance(obj, h5py.Group) and "ivals" in obj else None)
  return groups


def test_meson_output(tmp_path):
  rep = OperatorRepresentation(*[Operator(meson_field(dbar, u, g), P0) for g in MESON_VECTOR])
  path = tmp_path / "rho.h5"
  write_projected_operators_h5(rep, ("T1u",), str(path))
  groups = _operator_groups(path)
  assert set(groups) == {"meson_operators/dbaru/T1u_{}/SS_0".format(r) for r in (1, 2, 3)}
  for terms in groups.values():
    for coeff, (meson,) in terms:
      assert meson["flavors"] == ("dbar", "u")
      assert meson["momentum"] == (0, 0, 0)
  with h5py.File(path, "r") as h5:
    assert h5["meson_operators/dbaru/T1u_1/SS_0"].attrs["layout"] == "s1 s2 d1 d2"


def test_compact_tetraquark_output(tmp_path):
  tcc = DiquarkAntidiquark(c, c, ubar, dbar, P0, DIQUARK_AXIAL, ANTIDIQUARK_SCALAR,
                           channel_label="Tcc")
  path = tmp_path / "tcc.h5"
  tcc.write_h5(("T1g",), str(path))
  groups = _operator_groups(path)
  assert set(groups) == {"tetraquark_operators/Tcc/T1g_{}/SS_0".format(r) for r in (1, 2, 3)}
  for coeff, (tetraquark,) in groups["tetraquark_operators/Tcc/T1g_2/SS_0"]:
    assert tetraquark["flavors"] == ("c", "c", "ubar", "dbar")


def test_meson_meson_output(tmp_path):
  pipi = MesonMeson([(1, (dbar, u), (dbar, u))], MESON_PSEUDOSCALAR, MESON_PSEUDOSCALAR,
                    P0, [(0, 0), (1, 1)], channel_label="pipi_I2")
  path = tmp_path / "pipi.h5"
  pipi.write_h5(("A1g",), str(path))
  groups = _operator_groups(path)
  assert set(groups) == {"meson_meson_operators/pipi_I2/A1g_1/MH_0",
                         "meson_meson_operators/pipi_I2/A1g_1/MH_1"}
  for terms in groups.values():
    for coeff, hadrons in terms:
      assert len(hadrons) == 2
      assert all(h["flavors"] == ("dbar", "u") for h in hadrons)
      assert tuple(map(sum, zip(*(h["momentum"] for h in hadrons)))) == (0, 0, 0)


def test_dibaryon_output(tmp_path):
  alpha = DiracIdx("alpha_lambda")
  lam = baryon_field(s, u, d, alpha)
  dib = Dibaryon(two_baryon_spin_components(lam, lam, alpha), P0, [(0, 0)], SPIN_SINGLET,
                 channel_label="H_LL")
  path = tmp_path / "h.h5"
  dib.write_h5(("A1g",), str(path))
  groups = _operator_groups(path)
  assert set(groups) == {"dibaryon_operators/H_LL/A1g_1/MH_0"}
  for coeff, hadrons in groups["dibaryon_operators/H_LL/A1g_1/MH_0"]:
    assert [h["flavors"] for h in hadrons] == [("u", "d", "s"), ("u", "d", "s")]


def test_mixed_single_and_multi_hadron_operator_is_rejected():
  pi = Operator(meson_field(dbar, u, "g5"), P0)
  compact = Operator(diquark_antidiquark_field(c, c, ubar, dbar, "Cg1", "g5C"), P0)
  mixed = {**compact.coefficients, **(pi * pi).coefficients}
  with pytest.raises(NotImplementedError):
    operator_elementals(mixed)
