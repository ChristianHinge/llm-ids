"""Token-native ID codec.

Encodes a non-negative integer as a sequence of symbols. The first symbol
is a bare, standalone letters-only word (no prefix) -- chosen so a
token_id never starts with "." | "_" | "-", which would otherwise risk
being misread as a CLI flag, break DNS-label rules, or (for "-"
specifically) confuse shell tools like `rm`/`ls` that treat a leading "-"
as an option marker. Every symbol after the first is a single
"." | "_" | "-" prefix character followed by an ASCII-letters run, e.g.:

    Access.test-Node.report_ABC-xyz

Why this shape works: the tokenizer's own pretokenizer regex fuses exactly
one leading punctuation character onto a following run of letters into a
single pretoken *before* BPE merging ever runs -- so as long as each
"prefix+letters" (or, for the first symbol, bare "letters") string is
itself a listed vocab token, it always tokenizes as exactly one token,
regardless of what precedes or follows it. No separator character is ever
needed between symbols. See tests/test_id_codec.py for the empirical
verification (thousands of random symbol chains, checked against the real
tokenizer).

The two symbol pools are different sizes (bare words are the larger
pool), so this is a mixed-radix positional number system: the first
(most-significant) digit is drawn from `first_symbols` (radix
`first_radix`), every digit after it from `symbols` (radix `radix`).
"""

import json
import re
from pathlib import Path

from llm_ids.alphabets import Alphabet

DATA_DIR = Path(__file__).parent / "data"

_CACHE: dict[Alphabet, "AlphabetTable"] = {}

_FIRST_SYMBOL_RE = re.compile(r"^[A-Za-z]+")
_SYMBOL_RE = re.compile(r"[.\-_][A-Za-z]+")


class AlphabetTable:
    def __init__(self, name: str, first_symbols: list[str], symbols: list[str]):
        self.name = name
        self.first_symbols = first_symbols
        self.symbols = symbols
        self.first_radix = len(first_symbols)
        self.radix = len(symbols)
        self._first_index = {s: i for i, s in enumerate(first_symbols)}
        self._index = {s: i for i, s in enumerate(symbols)}


def load_alphabet(alphabet: Alphabet) -> AlphabetTable:
    if alphabet not in _CACHE:
        path = DATA_DIR / f"{alphabet.value}.json"
        raw = json.loads(path.read_text())
        _CACHE[alphabet] = AlphabetTable(
            name=raw["name"],
            first_symbols=raw["first_symbols"],
            symbols=raw["symbols"],
        )
    return _CACHE[alphabet]


def encode_int_compact(value: int, table: AlphabetTable) -> str:
    """Encode a non-negative integer as a token_id string, with no
    leading-zero-symbol padding: mixed-radix positional notation (the
    first symbol from `table.first_symbols`, every symbol after it from
    `table.symbols`), so smaller values produce shorter strings.

    Use this for values with real magnitude structure you want reflected
    in the output -- a counter, a content hash, a timestamp-derived id.
    """
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")

    r0, r1 = table.first_radix, table.radix
    if value < r0:
        return table.first_symbols[value]

    idxs = []
    v = value
    while v >= r0:
        v, idx = divmod(v, r1)
        idxs.append(idx)
    idxs.append(v)
    idxs.reverse()
    return table.first_symbols[idxs[0]] + "".join(table.symbols[i] for i in idxs[1:])


def decode_str(id_str: str, table: AlphabetTable) -> int:
    """Decode a token_id string back to its integer value.

    The first symbol is a bare letters-only run; every symbol after it
    starts with one of "." | "_" | "-", so the boundary between the first
    symbol and the rest, and between each symbol after that, is
    unambiguous with no separator needed."""
    if not id_str:
        raise ValueError("id must be non-empty")

    m = _FIRST_SYMBOL_RE.match(id_str)
    if not m:
        raise ValueError(f"id {id_str!r} must start with a letters-only symbol")
    first_sym = m.group()
    rest = id_str[len(first_sym):]

    rest_syms = _SYMBOL_RE.findall(rest)
    if "".join(rest_syms) != rest:
        raise ValueError(f"id {id_str!r} is not a bare word followed by "
                          f"[.\\-_][A-Za-z]+ symbols")

    try:
        value = table._first_index[first_sym]
    except KeyError as e:
        raise ValueError(f"invalid first symbol {e} in id {id_str!r}") from e

    for i, sym in enumerate(rest_syms):
        try:
            idx = table._index[sym]
        except KeyError as e:
            raise ValueError(f"invalid symbol {e} in id {id_str!r} at position {i + 1}") from e
        value = value * table.radix + idx
    return value
