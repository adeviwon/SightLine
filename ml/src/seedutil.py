"""
Process-stable seed derivation.

THE BUG THIS FIXES
------------------
Every corpus generator seeded itself from a profile NAME using:

    ds = seed * 7919 + abs(hash(name)) % 100003 + i

Python salts `str.__hash__` per process unless PYTHONHASHSEED is set, so
`hash("studio_clean")` returns a different integer in every interpreter. The
consequence was silent and total:

  * `bash run.sh eval` run three times produced three DIFFERENT corpora
  * `bash run.sh train-dncnn` produced a different training set each run
  * every measured number in the docs was unreproducible
  * evaluate.py's docstring nonetheless claimed "Every number is from a
    seeded, reproducible run" -- a claim that was simply false

Nothing crashed. That is what made it dangerous.

THE FIX
-------
`zlib.crc32` is a specified, stable checksum of the byte string. It does not
depend on the interpreter, the platform, or PYTHONHASHSEED, so the same
profile name always maps to the same seed on every machine forever.

`profile_seed()` is now the ONLY sanctioned way to derive a seed from a name.
`ml/src/repro_check.py` runs the derivation in two subprocesses with different
PYTHONHASHSEED values and fails if they disagree -- so this cannot regress
unnoticed.
"""

import zlib

__all__ = ["profile_seed", "name_hash", "SEEDER_ID"]

# Written into every generated artifact. Two corpora are only comparable if
# this string matches: the same seed under two different seeders selects
# different documents entirely, so "seed 123" alone does not identify a corpus.
# Bump it whenever corpus-generation semantics change.
SEEDER_ID = "zlib.crc32/seedutil-v1"


def name_hash(name: str) -> int:
    """
    Stable 31-bit hash of a string, identical across processes and platforms.

    `zlib.crc32` returns an unsigned 32-bit int; mask to 31 bits so the result
    is always a safe non-negative value on any platform, and so the result
    composes predictably with the caller's arithmetic.
    """
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def profile_seed(profile_name: str, base_seed: int) -> int:
    """
    Derive a document seed from a capture-profile name and a base seed.

    Replaces the old `abs(hash(name))` expression 1:1, so existing seed
    arithmetic is unchanged -- only the hash becomes stable.
    """
    return base_seed * 7919 + name_hash(profile_name) % 100003
