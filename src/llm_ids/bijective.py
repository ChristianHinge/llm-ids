"""Generic bijective numeration: encode/decode a variable-length sequence
of symbols (each in range 0..radix-1) as a single non-negative integer,
preserving the exact length -- including leading "zero" symbols -- unlike
plain positional notation, which loses that information (e.g. "007" and
"7" are both just 7 as an ordinary base-10 number).

Every length gets its own dedicated, non-overlapping range of integers:
length n has exactly radix**n possible sequences, so length n's range
starts right where every shorter length's range ends. Within a length, a
sequence's position is ordinary positional notation (most-significant
symbol first) -- unambiguous once the length itself is no longer in
question.

This is the engine behind char_token_id.py: radix=len(charset), symbols
are indices into a caller-supplied character set.
"""


def count_for_length(radix: int, n: int) -> int:
    return radix ** n


def offset_for_length(radix: int, n: int) -> int:
    """sum of counts for all lengths 1..n-1 -- the cumulative rank offset."""
    if n <= 0:
        return 0
    return (radix ** n - radix) // (radix - 1)


def capacity(radix: int, max_len: int) -> int:
    """total number of valid sequences of length 1..max_len."""
    return offset_for_length(radix, max_len + 1)


def sequence_to_int(indices: list[int], radix: int) -> int:
    """indices: symbol indices (each 0..radix-1), most significant first."""
    rank = 0
    for idx in indices:
        rank = rank * radix + idx
    return offset_for_length(radix, len(indices)) + rank


def int_to_sequence(value: int, radix: int, max_len: int) -> list[int]:
    cap = capacity(radix, max_len)
    if not (0 <= value < cap):
        raise ValueError(f"value out of range 0..{cap - 1}")
    n = 1
    while value >= count_for_length(radix, n):
        value -= count_for_length(radix, n)
        n += 1
    indices = [0] * n
    v = value
    for i in range(n - 1, -1, -1):
        v, indices[i] = divmod(v, radix)
    return indices
