"""
Reproducibility gate: `--seed` must produce the same corpus in a FRESH
process, every time.

Why this exists: every generator used `abs(hash(name))` to seed from a
profile name. Python salts `hash()` per process (PYTHONHASHSEED is random by
default), so three separate `bash run.sh eval` invocations silently built
three DIFFERENT corpora -- while evaluate.py's docstring claimed "Every number
is from a seeded, reproducible run". That claim was false.

This is the check that makes the claim true. It runs the seeding logic in two
subprocesses with different hash seeds and diffs the results; if the seeds
depend on the process, the corpora differ and this fails.

Run:  bash run.sh repro-check
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# The canonical derivation, exercised through the REAL entry point rather than
# a reimplementation. Step 1's snippet reproduces the old buggy expression to
# demonstrate the mechanism; step 2 must call the shipped code, or this test
# would pass while the actual generator was still broken.
SEED_SNIPPET = """
import sys, hashlib
sys.path.insert(0, "ml/src")
import capture
# This is the shipped code path: make_capture_set is what evaluate.py and
# train_restorer.py both build on, so if its output is stable, ours is.
#
# Hash the actual DEGRADED PIXELS, not the seed integers. Checking seeds would
# pass even if a profile's parameters were nondeterministic; hashing the
# rendered image catches that too. PIL output is a byte-exact function of the
# seed, so equal hashes mean genuinely identical corpora.
for rec in capture.make_capture_set(2, seed=123):
    h = hashlib.sha256(rec["degraded"].tobytes()).hexdigest()[:16]
    print(f'{rec["profile"]}:{rec["doc_type"]}:{h}')
"""

HASH_SNIPPET = """
print(hash("studio_clean"))
"""


def run_in_subprocess(script, hashseed):
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": str(hashseed),
           "HOME": "/tmp"}
    py = str(ROOT / ".venv" / "bin" / "python")
    r = subprocess.run([py, "-c", script], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()
    return r.stdout.strip(), None


def main():
    failures = []

    # Step 1: demonstrate the mechanism — hash() itself varies per process.
    # This is the ROOT CAUSE, shown in isolation.
    h1, e1 = run_in_subprocess(HASH_SNIPPET, 1)
    h2, e2 = run_in_subprocess(HASH_SNIPPET, 2)
    if e1 or e2:
        print(f"  could not probe hash(): {e1 or e2}")
        return 1
    if h1 == h2:
        print("  note: hash() was stable across these two seeds (unexpected)")
    else:
        print("  root cause confirmed: hash() is salted per process")
        print(f"    PYTHONHASHSEED=1 -> {h1}")
        print(f"    PYTHONHASHSEED=2 -> {h2}")

    # Step 2: does the SHIPPED corpus generator inherit that instability?
    c1, e1 = run_in_subprocess(SEED_SNIPPET, 1)
    c2, e2 = run_in_subprocess(SEED_SNIPPET, 2)
    if e1 or e2:
        print(f"  could not build corpora: {e1 or e2}")
        return 1

    a, b = c1.splitlines(), c2.splitlines()
    if c1 != c2:
        failures.append("capture.make_capture_set is not reproducible")
        print("\n  FAIL make_capture_set() differs between processes:")
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                print(f"    {a[i]}\n    {b[i]}")
    else:
        print(f"  OK   make_capture_set() identical across processes "
              f"({len(a)} samples, pixel-hashed)")

    # Step 3: the canonical, process-stable alternative must exist and be used.
    canon = """
import sys
sys.path.insert(0, "ml/src")
import capture
from seedutil import profile_seed
out = []
for name in capture.CAPTURE_PROFILES:
    out.append((name, profile_seed(name, 123)))
for n, s in out:
    print(f"{n}={s}")
"""
    p1, e1 = run_in_subprocess(canon, 1)
    p2, e2 = run_in_subprocess(canon, 2)
    if e1 or e2:
        print(f"  seedutil.profile_seed missing: {e1 or e2}")
        failures.append("seedutil.profile_seed not importable")
    elif p1 != p2:
        print("  FAIL seedutil.profile_seed is not stable")
        failures.append("seedutil.profile_seed varies")
    else:
        print("  OK   seedutil.profile_seed identical across processes")
        print("\n  canonical seeds:")
        for line in p1.splitlines():
            print(f"    {line}")

    print()
    if failures:
        print(f"REPRODUCIBILITY: FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        print("\nFix: replace abs(hash(name)) with seedutil.profile_seed(name, seed).")
        return 1
    print("REPRODUCIBILITY: PASSED — every number in the docs is re-derivable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
