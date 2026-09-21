"""llm_ids: encode data as token-efficient, LLM-friendly strings.

    from llm_ids import Alphabet, Codec

    counter_codec = Codec(Alphabet.O200K_V1, charset="0123456789")
    token_id = counter_codec.to_token_id(str(42))
    assert counter_codec.from_token_id(token_id) == "42"

    tracking_codec = Codec(Alphabet.O200K_V1, charset="0123456789")
    token_id = tracking_codec.to_token_id("00423817")
    assert tracking_codec.from_token_id(token_id) == "00423817"

    # opaque binary (hash digests, keys): hex-encode it, no dedicated
    # bytes mode needed. Hex text carries identical capacity to raw bytes.
    import hashlib
    hex_codec = Codec(Alphabet.O200K_V1, charset="0123456789abcdef")
    digest_hex = hashlib.sha256(b"hello").hexdigest()
    token_id = hex_codec.to_token_id(digest_hex)
    assert hex_codec.from_token_id(token_id) == digest_hex

See id_codec.py and char_token_id.py for the lower-level, alphabet-table-
based functions Codec wraps.
"""

from llm_ids.alphabets import Alphabet
from llm_ids.codec import Codec

__all__ = ["Alphabet", "Codec"]
