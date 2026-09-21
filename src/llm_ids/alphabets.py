"""Stable, library-owned identifiers for the alphabets llm_ids ships.

Each alphabet is a curated symbol set *we* chose -- which single-token
"prefix+letters" strings from a tokenizer's vocabulary count as valid, per
id_codec.py's rules -- not a passthrough of a third party's encoding name.
Pin one of these in your code instead of a bare tokenizer-encoding string:
we're free to revise which tokens an alphabet contains (e.g. to fix a
tokenizer quirk or trim an ambiguous case) and ship that as a new version
without changing a name you already depend on.

The `source_encoding` recorded in each alphabet's data file documents which
tokenizer vocabulary it was built from -- useful for picking the right
member for your target model family, and for regenerating the data (see
scripts/build_alphabets.py) -- but callers should depend on the Alphabet
name, not that provenance string.
"""

from enum import Enum


class Alphabet(Enum):
    """Compatible with tokenizers built on the matching source vocabulary
    (see each data file's "source_encoding"/"source_encodings"):

    O200K_V1: OpenAI's o200k_base vocabulary (GPT-4o, GPT-4.1, GPT-5 family,
    as of this writing).

    CL100K_V1: OpenAI's cl100k_base vocabulary (GPT-4, GPT-3.5-turbo, as of
    this writing).

    SHARED_V1: the intersection of o200k_base, cl100k_base, Llama 3.x, and
    Qwen3's prefix-word vocabularies -- a smaller alphabet (9737 symbols vs.
    10544-13031 solo) that produces token_ids guaranteed to cost one token
    per symbol on all four tokenizer families at once, for a small bits/token
    cost (~13.25 vs. 13.36-13.67 solo). Use this when the target model isn't
    known in advance, or spans more than one of those families; use a solo
    alphabet when you know exactly which one you're targeting.

    MISTRAL_TEKKEN_V1: Mistral's "Tekken" vocabulary (Mistral NeMo, Small/
    Large 2.x+, as of this writing). Its prefix-word vocabulary barely
    overlaps with the OpenAI/Llama/Qwen group, so it's not part of
    SHARED_V1 -- use this solo alphabet for Mistral-family targets.

    DEEPSEEK_V1: DeepSeek's vocabulary (DeepSeek-V3, R1, as of this
    writing). Same situation as Mistral: little overlap with the other
    families, so it gets its own solo alphabet.
    """

    O200K_V1 = "o200k-v1"
    CL100K_V1 = "cl100k-v1"
    SHARED_V1 = "shared-v1"
    MISTRAL_TEKKEN_V1 = "mistral-tekken-v1"
    DEEPSEEK_V1 = "deepseek-v1"
