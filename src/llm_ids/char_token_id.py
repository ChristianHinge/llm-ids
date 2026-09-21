"""Token-efficient token_id for a string over a caller-declared character
set (e.g. digits only, hex digits, alphanumeric).

A bijective-numeration codec (see bijective.py) parameterized by an
arbitrary caller-supplied alphabet: radix = len(charset). Use this for
data that's meaningfully a *string*, not a number -- a tracking number, a
confirmation code, a UUID's hex digits -- where a leading character equal
to charset[0] is part of the value ("00423817" is an 8-character code, not
the number 423817 with formatting stripped), not something to normalize
away.

Every function here takes an AlphabetTable explicitly -- no default
alphabet. See llm_ids.Codec for the recommended entry point.
"""

from llm_ids import bijective, id_codec
from llm_ids.id_codec import AlphabetTable

MAX_LEN = 1024


def validate_charset(charset: str) -> None:
    if len(charset) < 2:
        raise ValueError(f"charset must have at least 2 distinct characters, got {charset!r}")
    if len(set(charset)) != len(charset):
        raise ValueError(f"charset must not contain duplicate characters: {charset!r}")


def validate(data: str, charset: str) -> None:
    if not data or len(data) > MAX_LEN:
        raise ValueError(f"length must be 1..{MAX_LEN}, got {len(data)}")
    bad = set(data) - set(charset)
    if bad:
        raise ValueError(f"characters {sorted(bad)!r} not in charset {charset!r}")


def str_to_int(data: str, charset: str) -> int:
    validate(data, charset)
    index = {c: i for i, c in enumerate(charset)}
    indices = [index[c] for c in data]
    return bijective.sequence_to_int(indices, len(charset))


def int_to_str(value: int, charset: str, max_len: int = MAX_LEN) -> str:
    indices = bijective.int_to_sequence(value, len(charset), max_len)
    return "".join(charset[i] for i in indices)


def to_token_id(data: str, charset: str, table: AlphabetTable) -> str:
    """A string over `charset` -> token-efficient token_id string.

    Compact encoding: token_id length tracks the input's own rank/length.
    """
    value = str_to_int(data, charset)
    return id_codec.encode_int_compact(value, table)


def from_token_id(token_id: str, charset: str, table: AlphabetTable) -> str:
    """Token-efficient token_id string -> the exact original string."""
    value = id_codec.decode_str(token_id, table)
    return int_to_str(value, charset)
