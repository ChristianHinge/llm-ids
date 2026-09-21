"""Round-trip and leading-character-preservation tests for llm_ids.Codec /
char_token_id.

Run with: uv run pytest tests/test_char_token_id.py
"""

import secrets
import string

from llm_ids import Alphabet, Codec

TRIALS = 2000
DIGITS = "0123456789"
digit_codec = Codec(Alphabet.O200K_V1, charset=DIGITS)


def _random_digit_str() -> str:
    n = secrets.randbelow(20) + 1
    return "".join(secrets.choice(DIGITS) for _ in range(n))


def test_round_trip():
    for _ in range(TRIALS):
        data = _random_digit_str()
        token_id = digit_codec.to_token_id(data)
        assert digit_codec.from_token_id(token_id) == data, f"round trip failed for {data!r}"
    print(f"round-trip OK over {TRIALS} random digit strings")


def test_leading_zero_chars_preserved():
    """A tracking number like "00423817" is an 8-character code, not the
    number 423817 with formatting stripped -- the leading zeros must
    survive exactly."""
    cases = ["0", "007", "00423817", "0" * 32]
    for data in cases:
        token_id = digit_codec.to_token_id(data)
        decoded = digit_codec.from_token_id(token_id)
        assert decoded == data, f"lost leading zero chars: {data!r} -> {decoded!r}"
    print(f"[leading zeros] {len(cases)} cases round-tripped with exact value preserved")


def test_integers_round_trip_via_str():
    """Non-negative integers go through str() + a digit charset -- no
    dedicated integer mode. Small and large values both round-trip."""
    for value in [0, 1, 42, 10**6, secrets.randbits(256)]:
        token_id = digit_codec.to_token_id(str(value))
        assert int(digit_codec.from_token_id(token_id)) == value
    print("[integers via str] 0, 1, 42, 10**6, and a 256-bit value round-trip correctly")


def test_different_charsets_are_independent():
    alnum_codec = Codec(Alphabet.O200K_V1, charset=string.ascii_uppercase + DIGITS)
    hex_codec = Codec(Alphabet.O200K_V1, charset="0123456789abcdef")

    token_id = alnum_codec.to_token_id("A1B2C3")
    assert alnum_codec.from_token_id(token_id) == "A1B2C3"

    token_id = hex_codec.to_token_id("deadbeef")
    assert hex_codec.from_token_id(token_id) == "deadbeef"
    print("[independent charsets] alphanumeric and hex codecs round-trip independently")


def test_wrong_charset_does_not_raise_reliably():
    """Documents the known hazard: decoding with the wrong charset is not
    guaranteed to error, if the two charsets happen to overlap. This is
    why Codec requires an explicit charset with no default."""
    alnum_codec = Codec(Alphabet.O200K_V1, charset=string.ascii_uppercase + DIGITS)
    data = "A1B2C3D4E5"
    token_id = digit_codec.to_token_id("1234567890")
    try:
        result = alnum_codec.from_token_id(token_id)
    except ValueError:
        return  # also an acceptable outcome -- just not a guaranteed one
    assert result != "1234567890"
    print("[cross-charset] confirmed: wrong Codec decoded without raising, "
          "and produced a different string (expected hazard, not a bug)")


def test_rejects_invalid_charset():
    bad_charsets = ["", "a", "aab"]  # empty, too short, duplicate character
    for charset in bad_charsets:
        try:
            Codec(Alphabet.O200K_V1, charset=charset)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for charset {charset!r}, but it succeeded")
    print("[invalid charset] empty/too-short/duplicate charsets correctly rejected")


def test_rejects_characters_outside_charset():
    try:
        digit_codec.to_token_id("12a34")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError encoding a character outside the charset")
    print("[out-of-charset] character outside the declared charset correctly rejected")


def test_rejects_invalid_input():
    bad = ["", "0" * 1025]
    for data in bad:
        try:
            digit_codec.to_token_id(data)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError encoding {data!r}, but it succeeded")
    print("[invalid input] empty and over-length inputs correctly rejected")


def test_rejects_wrong_type():
    for value in [b"not a str", 42, ["a", "list"]]:
        try:
            digit_codec.to_token_id(value)
        except TypeError:
            continue
        raise AssertionError(f"expected TypeError encoding {value!r}, but it succeeded")
    print("[wrong type] non-str input correctly rejected with TypeError")


def test_repr():
    assert repr(digit_codec) == f"Codec({Alphabet.O200K_V1}, charset={DIGITS!r})"
    print("[repr] Codec repr shows alphabet and charset")


if __name__ == "__main__":
    test_round_trip()
    test_leading_zero_chars_preserved()
    test_integers_round_trip_via_str()
    test_different_charsets_are_independent()
    test_wrong_charset_does_not_raise_reliably()
    test_rejects_invalid_charset()
    test_rejects_characters_outside_charset()
    test_rejects_invalid_input()
    test_rejects_wrong_type()
    test_repr()
    print("\nAll tests passed.")
