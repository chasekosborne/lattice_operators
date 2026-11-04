"""Partner rows must form multiplets that transform with the tabulated irrep matrices."""

import numpy as np
import pytest
from sympy import Eijk, Matrix

from context import operators  # noqa: F401
from operators.cubic_rotations import E, Momentum, P0
from operators.operators import (
    TRANSPOSED_IRREP_TABLES,
    AntiQuarkField,
    ColorIdx,
    DiracIdx,
    Operator,
    OperatorRepresentation,
    QuarkField,
)
from operators.tetraquark import ANTIDIQUARK_AXIAL, DIQUARK_AXIAL, DiquarkAntidiquark


def _num(m):
  return np.array(Matrix(m).evalf(), dtype=complex)


def _assert_multiplets_transform(rep, irrep, acc):
  dim = int(rep.little_group.getCharacter(irrep, E))
  first = rep.getLinearlyIndependentProjectedCoefficientRows(irrep, row=1, irrep_matrices=acc)
  assert first
  for src in first:
    V = [_num(src).flatten()]
    for row in range(2, dim + 1):
      V.append(_num(rep.getPartnerRowCoefficientRows(
          irrep, [src], row, 1, irrep_matrices=acc)[0]).flatten())
    V = np.array(V).T
    assert all(np.linalg.norm(V[:, n]) > 1e-8 for n in range(dim))
    for R in rep.little_group.elements:
      gamma = _num(acc(irrep, R))
      if irrep in TRANSPOSED_IRREP_TABLES:
        gamma = gamma.T
      assert np.allclose(_num(rep.getRepresentationMatrix(R)) @ V, V @ gamma)


def _delta(momentum):
  u = QuarkField.create("u")
  a, b, c = ColorIdx("a"), ColorIdx("b"), ColorIdx("c")
  i, j, k = DiracIdx("i"), DiracIdx("j"), DiracIdx("k")
  delta = Eijk(a, b, c) * u[a, i] * u[b, j] * u[c, k]
  return OperatorRepresentation(*[
      Operator(delta.subs({i: x, j: y, k: z}), momentum)
      for x in range(4) for y in range(x, 4) for z in range(y, 4)])


@pytest.mark.parametrize("momentum, irrep", [
    (P0, "G1g"),
    (P0, "Hg"),
    (Momentum([0, 0, 1]), "G1"),
    (Momentum([0, 0, 1]), "G2"),
    (Momentum([1, 1, 1]), "G"),
])
def test_fermionic_partner_rows(momentum, irrep):
  rep = _delta(momentum)
  _assert_multiplets_transform(rep, irrep, rep.getIrrepAccessor(include_odd_parity=True))


@pytest.mark.parametrize("momentum, irrep", [
    (P0, "Eg"),
    (P0, "T1g"),
    (P0, "T2g"),
    (Momentum([0, 0, 1]), "E"),
    (Momentum([1, 1, 1]), "E"),
])
def test_bosonic_partner_rows(momentum, irrep):
  c = QuarkField.create("c")
  tcc = DiquarkAntidiquark(c, c, AntiQuarkField.create("u"), AntiQuarkField.create("d"),
                           momentum, DIQUARK_AXIAL, ANTIDIQUARK_AXIAL)
  _assert_multiplets_transform(tcc.representation, irrep, tcc.get_irrep_accessor())
