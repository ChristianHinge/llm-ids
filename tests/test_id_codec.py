"""Round-trip and real-tokenizer stability tests for llm_ids.id_codec.

Run with: uv run pytest tests/test_id_codec.py
"""

import re
import secrets

import tiktoken

from llm_ids import id_codec
from llm_ids.alphabets import Alphabet

TRIALS = 5000
ALPHABETS = list(Alphabet)
# real tiktoken encodings to verify each alphabet's symbols against. Only
# tiktoken-family alphabets are checked here. scripts/build_alphabets.py
# additionally verifies SHARED_V1 against Llama 3.1 and Qwen3, and
# MISTRAL_TEKKEN_V1/DEEPSEEK_V1 against their own (non-tiktoken) source
# tokenizers, at build time. Those aren't re-checked here to keep the
# routine test suite free of gated/network dependencies -- the checked-in
# JSON is trusted between rebuilds.
STABILITY_ENCODINGS = {
    Alphabet.O200K_V1: ["o200k_base"],
    Alphabet.CL100K_V1: ["cl100k_base"],
    Alphabet.SHARED_V1: ["o200k_base", "cl100k_base"],
}

_SYMBOL_COUNT_RE = re.compile(r"[.\-_][A-Za-z]+")


def _symbol_count(id_str: str) -> int:
    """How many symbols id_str decodes to: 1 for the bare first word, plus
    one per [.\\-_][A-Za-z]+ run after it."""
    m = re.match(r"^[A-Za-z]+", id_str)
    rest = id_str[len(m.group()):]
    return 1 + len(_SYMBOL_COUNT_RE.findall(rest))


def test_round_trip():
    for alphabet in ALPHABETS:
        table = id_codec.load_alphabet(alphabet)
        for _ in range(2000):
            v = secrets.randbits(256)
            s = id_codec.encode_int_compact(v, table)
            assert id_codec.decode_str(s, table) == v, f"round trip failed: {v} -> {s!r}"
        for v in [0, table.first_radix - 1, table.first_radix]:
            s = id_codec.encode_int_compact(v, table)
            assert id_codec.decode_str(s, table) == v
        print(f"[{alphabet}] round-trip OK over 2003 random/boundary values")


def test_first_symbol_is_always_bare():
    """The actual point of this design: a token_id never starts with
    "." | "_" | "-"."""
    for alphabet in ALPHABETS:
        table = id_codec.load_alphabet(alphabet)
        for v in [0, 1, table.first_radix - 1, table.first_radix,
                  table.first_radix * table.radix, secrets.randbits(256)]:
            s = id_codec.encode_int_compact(v, table)
            assert s[0].isalpha(), f"[{alphabet}] token_id {s!r} for value {v} starts with {s[0]!r}"
    print("[bare first symbol] every generated token_id starts with a plain letter")


def test_compact_round_trip():
    for alphabet in ALPHABETS:
        table = id_codec.load_alphabet(alphabet)
        seen = set()
        for v in range(200_000):
            s = id_codec.encode_int_compact(v, table)
            assert id_codec.decode_str(s, table) == v
            assert s not in seen, f"collision at {v}: {s!r}"
            seen.add(s)
        print(f"[{alphabet}] compact round-trip OK, 200000 sequential values collision-free")


def test_real_tokenizer_stability():
    """The actual claim under test: a real token_id string, on its own,
    always costs exactly one token per symbol -- never more, because every
    symbol (the bare first word, and every connector-prefixed symbol after
    it) is guaranteed to be exactly one token."""
    for alphabet, encoding_names in STABILITY_ENCODINGS.items():
        table = id_codec.load_alphabet(alphabet)

        for encoding_name in encoding_names:
            enc = tiktoken.get_encoding(encoding_name)

            mismatches = []
            for _ in range(TRIALS):
                v = secrets.randbits(256)
                id_str = id_codec.encode_int_compact(v, table)
                expected = _symbol_count(id_str)
                actual = len(enc.encode(id_str))
                if actual != expected:
                    mismatches.append((id_str, expected, actual))

            rate = 100 * len(mismatches) / TRIALS
            print(f"[{alphabet} vs {encoding_name}] bare-ID stability: "
                  f"{len(mismatches)}/{TRIALS} mismatches ({rate:.2f}%)")
            for s, expected, actual in mismatches[:3]:
                print(f"    mismatch: {s!r} -> expected {expected}, got {actual} tokens")
            assert not mismatches, f"{alphabet} vs {encoding_name}: token-count mismatches found"


def test_malformed_input_rejected():
    """decode_str should raise, not misbehave, on bad input."""
    table = id_codec.load_alphabet(Alphabet.O200K_V1)
    bad_inputs = [
        "",                     # empty
        ".",                    # no bare first symbol at all
        ".Access",              # starts with a connector char, not a bare word
        "Access1",              # digit breaks the first symbol, "1" left dangling
        "Access .test",         # contains a space
        "notarealbareword.test",  # bare word not in first_symbols
        "Access.notarealword",  # connector+letters not in symbols
    ]
    for s in bad_inputs:
        try:
            id_codec.decode_str(s, table)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError decoding {s!r}, but it succeeded")
    print("[malformed input] all bad inputs correctly rejected with ValueError")


if __name__ == "__main__":
    test_round_trip()
    test_first_symbol_is_always_bare()
    test_compact_round_trip()
    test_real_tokenizer_stability()
    test_malformed_input_rejected()
    print("\nAll tests passed.")
