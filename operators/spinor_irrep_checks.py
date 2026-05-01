# sanity checks for spinor irreps

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from sympy import Matrix, S, eye, simplify, trace


@dataclass
class SpinorIrrepVerification:
  """Returned when all checks pass."""

  little_group_order: int
  representation_dimension: int
  homomorphism_checks: int


def verify_extracted_spinor_irrep(
    little_group,
    irrep_label: str,
    rep_by_rotation: dict,
    *,
    homomorphism_sample_pairs: int = 256,
    homomorphism_rng: random.Random | None = None,
):
  """Validate traces vs characters and (sampled) homomorphism law ``D(R)D(S)=D(RS)``."""
  elems = frozenset(little_group.elements)
  keys = set(rep_by_rotation.keys())
  if keys != elems:
    raise ValueError(
        "Keys must match LittleGroup.elements exactly (missing {}, extra {})".format(
            sorted(elems - keys, key=repr),
            sorted(keys - elems, key=repr),
        )
    )

  from operators import cubic_rotations as cr

  identity = cr.E
  if identity not in elems:
    raise ValueError("little_group.elements does not contain identity E")

  any_rot = next(iter(elems))
  dim = rep_by_rotation[any_rot].rows
  if rep_by_rotation[any_rot].cols != dim:
    raise ValueError("Representation matrices must be square")

  id_mat = simplify(rep_by_rotation[identity])
  if id_mat != eye(dim):
    raise ValueError("D(E) != I (got {})".format(id_mat))

  for R in elems:
    M = rep_by_rotation[R]
    if M.rows != dim or M.cols != dim:
      raise ValueError("Inconsistent dimensions at {}".format(repr(R)))

  for R in elems:
    chi_rep = simplify(trace(rep_by_rotation[R]))
    chi_tab = little_group.getCharacter(irrep_label, R)
    delta = simplify(chi_rep - chi_tab)
    if delta != S.Zero and delta != 0:
      raise ValueError(
          "Character mismatch for {!r} at {}: trace(rep)={} table={}".format(
              irrep_label, repr(R), chi_rep, chi_tab
          )
      )

  elem_list = sorted(elems, key=repr)
  n = len(elem_list)
  if homomorphism_sample_pairs <= 0 or n == 0:
    return SpinorIrrepVerification(
        little_group_order=n,
        representation_dimension=dim,
        homomorphism_checks=0,
    )

  rng = homomorphism_rng if homomorphism_rng is not None else random.Random()
  pairs_idx = list(itertools.product(range(n), repeat=2))
  target = min(homomorphism_sample_pairs, len(pairs_idx))
  if target == len(pairs_idx):
    chosen = pairs_idx
  else:
    chosen = rng.sample(pairs_idx, target)

  zeros = Matrix.zeros(dim, dim)
  for i, j in chosen:
    R = elem_list[i]
    Srot = elem_list[j]
    RS = R * Srot
    if RS is None:
      raise ValueError("CubicRotation product undefined: {!r} * {!r}".format(R, Srot))
    if RS not in elems:
      raise ValueError(
          "Product leaves little group: {!r} * {!r} = {!r}".format(R, Srot, RS)
      )

    lhs = rep_by_rotation[R] * rep_by_rotation[Srot]
    rhs = rep_by_rotation[RS]
    diff = (lhs - rhs).applyfunc(simplify)
    if diff != zeros:
      raise ValueError(
          "Homomorphism failed for {!r} * {!r} = {!r}: D(R)D(S) != D(RS)".format(
              R, Srot, RS
          )
      )

  return SpinorIrrepVerification(
      little_group_order=n,
      representation_dimension=dim,
      homomorphism_checks=len(chosen),
  )


def matrices_from_hardcoded_strings(rep_str_dict: dict[str, list]) -> dict:
  """Convert one hardcoded irrep entry (string keys) to rotation -> Matrix."""
  from operators import cubic_rotations as cr

  repr_map = {repr(R): R for R in cr._POINT_GROUP}
  out = {}
  for rot_key, rows in rep_str_dict.items():
    if rot_key in cr._POINT_GROUP_NAME_TO_ROTATION:
      R = cr._POINT_GROUP_NAME_TO_ROTATION[rot_key]
    elif rot_key in repr_map:
      R = repr_map[rot_key]
    else:
      raise ValueError("Unknown rotation key {!r}".format(rot_key))
    out[R] = Matrix(rows)
  return out


def first_momentum_for_lg_irrep(lg_name: str, irrep: str):
  """Return a template momentum for ``(lg_name, irrep)`` (first match)."""
  from operators import cubic_rotations as cr

  for mom, labels in cr._FERMIONIC_LITTLE_GROUP_IRREPS.items():
    if cr._LITTLE_GROUPS[mom.reduced_pref] != lg_name:
      continue
    if irrep not in labels:
      continue
    return mom
  raise KeyError("No fermionic template for ({}, {})".format(lg_name, irrep))
