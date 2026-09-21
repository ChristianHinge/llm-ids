"""Codec: the recommended entry point for llm_ids.

Binds one Alphabet (the output token vocabulary) and one character set at
construction. Both are fixed for the life of the object, so
to_token_id/from_token_id never take extra arguments, and there's no way
to accidentally decode with the wrong charset -- construct one Codec per
charset you use and reuse it.

Non-negative integers use this too: `str(value)` and a digit charset
produce exactly the same token count as encoding the integer's magnitude
directly would, so there's no separate integer-shaped mode to maintain.

Opaque binary data (hash digests, encryption keys) has no dedicated mode
either: convert it to hex text (`data.hex()`) and use a hex charset --
that's a lossless, zero-ambiguity, zero-efficiency-cost conversion (N
bytes and 2N hex characters both cover exactly 2**(8*N) possible values).
"""

from llm_ids import char_token_id, id_codec
from llm_ids.alphabets import Alphabet


class Codec:
    def __init__(self, alphabet: Alphabet, charset: str):
        char_token_id.validate_charset(charset)
        self.alphabet = alphabet
        self._table = id_codec.load_alphabet(alphabet)
        self._charset = charset

    def to_token_id(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Codec.to_token_id requires a str, got {type(value).__name__}")
        return char_token_id.to_token_id(value, self._charset, self._table)

    def from_token_id(self, token_id: str) -> str:
        return char_token_id.from_token_id(token_id, self._charset, self._table)

    def __repr__(self) -> str:
        return f"Codec({self.alphabet}, charset={self._charset!r})"
