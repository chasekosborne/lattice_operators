from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path():
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_hardcoded(path: Path) -> dict:
  spec = importlib.util.spec_from_file_location("_hc_spinor_verify", path)
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(mod)
  return dict(mod.HARD_CODED_SPINOR_IRREP_STR_MATRICES)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--path",
      type=Path,
      default=REPO_ROOT / "operators" / "hardcoded_spinor_irreps.py",
      help="Module containing HARD_CODED_SPINOR_IRREP_STR_MATRICES",
  )
  parser.add_argument(
      "--pairs",
      type=int,
      default=384,
      metavar="K",
      help="Homomorphism samples per irrep (0 = trace/identity only)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="RNG seed for homomorphism sampling",
  )
  args = parser.parse_args()

  _ensure_repo_on_path()

  from operators import cubic_rotations as cr
  from operators.spinor_irrep_checks import (
      first_momentum_for_lg_irrep,
      matrices_from_hardcoded_strings,
      verify_extracted_spinor_irrep,
  )

  path = args.path.resolve()
  data = load_hardcoded(path)
  if not data:
    print("No irreps in {}; nothing to do.".format(path), file=sys.stderr)
    return 0

  rng_master = random.Random(args.seed)
  failures = 0

  for (lg_name, irrep) in sorted(data.keys(), key=lambda t: (t[0], t[1])):
    try:
      mom = first_momentum_for_lg_irrep(lg_name, irrep)
    except KeyError as exc:
      print("SKIP ({}, {}): {}".format(lg_name, irrep, exc), file=sys.stderr)
      failures += 1
      continue

    lg = cr.LittleGroup(False, mom)
    rep = matrices_from_hardcoded_strings(data[(lg_name, irrep)])
    try:
      summary = verify_extracted_spinor_irrep(
          lg,
          irrep,
          rep,
          homomorphism_sample_pairs=max(0, args.pairs),
          homomorphism_rng=random.Random(rng_master.randint(0, 2**30 - 1)),
      )
    except ValueError as exc:
      print("FAIL ({}, {}): {}".format(lg_name, irrep, exc), file=sys.stderr)
      failures += 1
      continue

    print(
        "OK  ({}, {})  dim={} |G|={} hom_samples={}".format(
            lg_name,
            irrep,
            summary.representation_dimension,
            summary.little_group_order,
            summary.homomorphism_checks,
        ),
        file=sys.stderr,
    )

  if failures:
    print("{} block(s) failed or skipped.".format(failures), file=sys.stderr)
    return 1
  print("All {} block(s) passed.".format(len(data)), file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
