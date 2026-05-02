from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path():
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def count_target_irreps(include_odd_parity: bool) -> int:
  _ensure_repo_on_path()
  from operators import cubic_rotations as cr

  n = 0
  for _, irreps in cr._FERMIONIC_LITTLE_GROUP_IRREPS.items():
    for irrep in irreps:
      if irrep in cr._FERMIONIC_SPINOR_U_LABELS and not include_odd_parity:
        continue
      n += 1
  return n


def load_existing_hardcoded(path: Path) -> dict:
  if not path.is_file():
    return {}
  spec = importlib.util.spec_from_file_location("_hardcoded_spinor_resume", path)
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(mod)
  raw = getattr(mod, "HARD_CODED_SPINOR_IRREP_STR_MATRICES", None)
  if raw is None:
    return {}
  return dict(raw)


def canonical_hardcoded_accumulated(accumulated: dict, cr) -> dict:
  repr_to_name = {repr(R): cr._POINT_GROUP_NAMES[R] for R in cr._POINT_GROUP}
  out = {}
  for irrep_key, block in accumulated.items():
    new_block = {}
    for k, rows in block.items():
      if k in cr._POINT_GROUP_NAME_TO_ROTATION:
        nk = k
      elif k in repr_to_name:
        nk = repr_to_name[k]
      else:
        raise ValueError(
            "Unknown rotation key {!r} while resuming {!r}".format(k, irrep_key)
        )
      new_block[nk] = rows
    out[irrep_key] = new_block
  return out


def _serialize_matrix_rows(rows) -> str:
  """Render nested lists of sympy exprs as a quoted-string Python literal.

  Each element is serialised as ``repr(str(elem))`` — i.e. a quoted sympy
  string such as ``'1/2 + I/2'``.  On load, ``sympify`` parses ``'1/2'``
  symbolically as ``Rational(1, 2)``, so the round-trip is exact with no
  floating-point contamination and the file stays human-readable.
  """
  return (
      "["
      + ", ".join(
          "[" + ", ".join(repr(str(elem)) for elem in row) + "]" for row in rows
      )
      + "]"
  )


def rep_to_serializable(extracted: dict, cr) -> dict[str, list]:
  """Map condensed rotation symbol -> nested lists (see ``_POINT_GROUP_NAMES``)."""
  names = cr._POINT_GROUP_NAMES
  out = {}
  for R, M in extracted.items():
    out[names[R]] = M.tolist()
  return out


def write_hardcoded_module(path: Path, accumulated: dict) -> None:
  """Emit a loadable module matching operators/hardcoded_spinor_irreps format."""
  header = '''"""Auto-generated hardcoded spinor irrep matrices.

Built by scripts/build_hardcoded_spinor_irreps_from_cubic.py using
operators.cubic_rotations.iter_spinor_irrep_matrix_blocks (Oh via
get_spinor_irrep_matrix; other little groups via spin-j extraction).
Do not hand-edit unless you know what you are doing.

Rotation keys are condensed symbols (E, C2x, …, I_C4zi) matching
operators.cubic_rotations._POINT_GROUP_NAME_TO_ROTATION; legacy repr() keys are
still accepted on load.
"""

from sympy import *  # noqa: F401,F403

HARD_CODED_SPINOR_IRREP_STR_MATRICES = {
'''
  lines = [header.rstrip("\n")]
  for (lg, irrep) in sorted(accumulated.keys(), key=lambda t: (t[0], t[1])):
    rep = accumulated[(lg, irrep)]
    lines.append(f"    ({lg!r}, {irrep!r}): {{")
    for rot_repr in sorted(rep.keys()):
      rows = rep[rot_repr]
      lines.append(f"        {rot_repr!r}: {_serialize_matrix_rows(rows)},")
    lines.append("    },")
  lines.append("}\n")

  text = "\n".join(lines)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(text)
  tmp.replace(path)


def _format_eta(seconds: float) -> str:
  if seconds != seconds or seconds < 0:
    return "?"
  return str(timedelta(seconds=int(seconds)))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--output",
      type=Path,
      default=REPO_ROOT / "operators" / "hardcoded_spinor_irreps.py",
      help="Path to the Python module to write (default: operators/hardcoded_spinor_irreps.py)",
  )
  parser.add_argument(
      "--include-odd-parity",
      action="store_true",
      help="Include odd-parity (u) Oh irreps as well as even (g).",
  )
  parser.add_argument(
      "--resume",
      action="store_true",
      help="Skip irreps already present in the output file.",
  )
  parser.add_argument(
      "--limit",
      type=int,
      default=None,
      metavar="N",
      help="Stop after N newly computed irreps (for smoke tests).",
  )
  parser.add_argument(
      "--verify",
      action="store_true",
      help="Run post-extraction checks (characters + sampled homomorphism law).",
  )
  parser.add_argument(
      "--verify-pairs",
      type=int,
      default=256,
      metavar="K",
      help="Number of random (R,S) pairs for D(R)D(S)=D(RS) check (0=characters only).",
  )
  parser.add_argument(
      "--verify-seed",
      type=int,
      default=None,
      help="RNG seed for homomorphism sampling (default: nondeterministic).",
  )
  args = parser.parse_args()

  _ensure_repo_on_path()

  from operators import cubic_rotations as cr
  from operators.spinor_irrep_checks import verify_extracted_spinor_irrep

  output_path = args.output.resolve()
  total_targets = count_target_irreps(args.include_odd_parity)

  accumulated = {}
  if args.resume:
    accumulated = load_existing_hardcoded(output_path)

  if accumulated:
    accumulated = canonical_hardcoded_accumulated(accumulated, cr)

  script_t0 = time.perf_counter()
  done_this_session = 0
  skipped = 0

  print(
      f"Targets: {total_targets} irrep blocks "
      f"({'with' if args.include_odd_parity else 'without'} odd-parity Oh irreps). "
      f"Already on disk: {len(accumulated)}.",
      file=sys.stderr,
      flush=True,
  )
  if not args.verify:
    print("Verification disabled (pass --verify to enable).", file=sys.stderr, flush=True)
  else:
    print(
        "Verification: characters + identity; "
        "homomorphism sample size {}.".format(args.verify_pairs),
        file=sys.stderr,
        flush=True,
    )

  iterator = cr.iter_spinor_irrep_matrix_blocks(
      include_odd_parity=args.include_odd_parity
  )

  verify_rng = random.Random(args.verify_seed) if args.verify_seed is not None else None

  while True:
    step_t0 = time.perf_counter()
    try:
      key, extracted, momentum = next(iterator)
    except StopIteration:
      break
    extract_dt = time.perf_counter() - step_t0

    if key in accumulated:
      skipped += 1
      continue

    verify_dt = 0.0
    verify_report = ""
    if args.verify:
      v0 = time.perf_counter()
      lg = cr.LittleGroup(False, momentum)
      summary = verify_extracted_spinor_irrep(
          lg,
          key[1],
          extracted,
          homomorphism_sample_pairs=max(0, args.verify_pairs),
          homomorphism_rng=verify_rng,
      )
      verify_dt = time.perf_counter() - v0
      verify_report = (
          " verify_ok dim={} |G|={} hom_pairs={} ({:.2f}s)".format(
              summary.representation_dimension,
              summary.little_group_order,
              summary.homomorphism_checks,
              verify_dt,
          )
      )

    accumulated[key] = rep_to_serializable(extracted, cr)
    write_hardcoded_module(output_path, accumulated)
    total_dt = time.perf_counter() - step_t0

    done_this_session += 1
    elapsed = time.perf_counter() - script_t0
    done_total = len(accumulated)
    remaining = max(0, total_targets - done_total)
    eta_sec = (elapsed / done_this_session) * remaining if done_this_session else float("nan")

    print(
        f"[{done_total}/{total_targets}] checkpoint {key!r} | "
        f"extract={extract_dt:.2f}s total_inc_write={total_dt:.2f}s | "
        f"session_elapsed={elapsed:.1f}s | ETA≈{_format_eta(eta_sec)}"
        f"{verify_report}",
        file=sys.stderr,
        flush=True,
    )

    if args.limit is not None and done_this_session >= args.limit:
      print(f"--limit {args.limit} reached; stopping.", file=sys.stderr, flush=True)
      break

  done_total = len(accumulated)
  if args.limit is not None:
    print(
        f"Stopped after --limit: {done_total}/{total_targets} irreps in {output_path}. "
        f"Resume with --resume when ready.",
        file=sys.stderr,
        flush=True,
    )
  elif done_total < total_targets:
    print(
        f"Incomplete: {done_total}/{total_targets} irreps in {output_path}. "
        f"Use --resume to continue (built {done_this_session} this session, "
        f"skipped {skipped} already present).",
        file=sys.stderr,
        flush=True,
    )
  else:
    print(
        f"Complete: {done_total}/{total_targets} irreps in {output_path}. "
        f"Built {done_this_session} this session, skipped {skipped} resume hits.",
        file=sys.stderr,
        flush=True,
    )

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
