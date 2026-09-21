"""One-time generator for the token-native alphabets used by id_codec.py.

Each alphabet has two symbol pools:

- `first_symbols`: standalone letters-only words (e.g. "Access"), used
  only for a token_id's first symbol, so the output never starts with
  "." | "_" | "-" (which risks being misread as a CLI flag, breaks
  DNS-label rules, or confuses shell tools that treat a leading "-" as an
  option marker).
- `symbols`: a single "." | "_" | "-" prefix character followed by one or
  more ASCII letters (e.g. ".Access", "_report", "-Node"), used for every
  symbol after the first.

BPE tokenizers of the "tiktoken family" (OpenAI, Llama 3.x, Qwen3, and
others) fuse exactly one leading punctuation character onto a following
run of letters as a single pretoken *before* merging ever runs -- so any
string of that shape which is itself a listed vocab token is guaranteed to
tokenize as exactly one token, chained back to back with no separator
needed between symbols (verified empirically against each real tokenizer
below, and in tests/test_id_codec.py).

Each alphabet built here corresponds to one llm_ids.Alphabet member --
that's llm_ids' own name for the symbol set, not any source tokenizer's
encoding name (see src/llm_ids/alphabets.py for why). "source_encoding"/
"source_encodings" is recorded in the output purely as provenance for
regenerating the data.

Solo alphabets (O200K_V1, CL100K_V1, ...) are every qualifying symbol in
one tokenizer's vocab. SHARED_V1 is the *intersection* of several
tokenizers' qualifying symbols -- a smaller alphabet, but one that
produces token_ids guaranteed to cost one token per symbol on every
tokenizer it's built from at once. Building it re-verifies the fusion
property empirically against each real tokenizer (not just against the
vocab list), since a shared symbol set is only useful if it's actually
safe on every member.

Run this whenever you want to (re)generate an Alphabet member's data file.
Not needed at runtime by consumers of the library -- the JSON is checked
in. This script's dependencies (tiktoken, tokenizers, huggingface_hub) are
dev-only extras. Building anything sourced from the Hugging Face Hub
downloads a tokenizer.json file (small; no model weights) and, for gated
repos (currently just meta-llama/Llama-3.1-8B), requires being logged in
with a Hugging Face account that has accepted that repo's license.
"""

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tiktoken
import tokenizers
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm_ids.alphabets import Alphabet  # noqa: E402

PREFIXES = ".-_"
FIRST_SYMBOL_RE = re.compile(r"^[A-Za-z]+$")
SYMBOL_RE = re.compile(r"^[" + re.escape(PREFIXES) + r"][A-Za-z]+$")
DATA_DIR = Path(__file__).parent.parent / "src" / "llm_ids" / "data"

# which tokenizer each solo Alphabet member is built from -- a bare name is
# resolved via tiktoken, anything containing "/" via the HF Hub
SOLO_SOURCES = {
    Alphabet.O200K_V1: "o200k_base",
    Alphabet.CL100K_V1: "cl100k_base",
    Alphabet.MISTRAL_TEKKEN_V1: "mistralai/Mistral-Small-24B-Instruct-2501",
    Alphabet.DEEPSEEK_V1: "deepseek-ai/DeepSeek-V3",
}

# which tokenizers SHARED_V1 is the intersection of
SHARED_SOURCES = ["o200k_base", "cl100k_base", "meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B"]


@dataclass
class TokenizerSource:
    name: str
    vocab_strs: Callable[[], set[str]]
    token_count: Callable[[str], int]


def _tiktoken_source(encoding_name: str) -> TokenizerSource:
    enc = tiktoken.get_encoding(encoding_name)

    def vocab_strs() -> set[str]:
        strs = set()
        for tok_bytes in enc._mergeable_ranks.keys():
            try:
                strs.add(tok_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                pass
        return strs

    return TokenizerSource(encoding_name, vocab_strs, lambda s: len(enc.encode(s)))


def _hf_source(repo_id: str) -> TokenizerSource:
    path = hf_hub_download(repo_id, filename="tokenizer.json")
    tok = tokenizers.Tokenizer.from_file(path)
    return TokenizerSource(repo_id, lambda: set(tok.get_vocab().keys()),
                            lambda s: len(tok.encode(s, add_special_tokens=False).ids))


def _source(name: str) -> TokenizerSource:
    return _hf_source(name) if "/" in name else _tiktoken_source(name)


def _filtered_symbols(source: TokenizerSource) -> set[str]:
    return {s for s in source.vocab_strs() if SYMBOL_RE.match(s)}


def _filtered_first_symbols(source: TokenizerSource) -> set[str]:
    return {s for s in source.vocab_strs() if FIRST_SYMBOL_RE.match(s)}


def _verify_fusion(first_symbols: list[str], symbols: list[str], source: TokenizerSource,
                    trials: int = 2000, chain_len: int = 18, seed: int = 0) -> None:
    """Chain a random first symbol plus random connector-prefixed symbols,
    with no separator, and confirm the real tokenizer still costs exactly
    one token per symbol."""
    rng = random.Random(seed)
    mismatches = 0
    for _ in range(trials):
        chosen = [rng.choice(first_symbols)] + [rng.choice(symbols) for _ in range(chain_len - 1)]
        if source.token_count("".join(chosen)) != chain_len:
            mismatches += 1
    if mismatches:
        raise RuntimeError(f"fusion check failed for {source.name}: "
                            f"{mismatches}/{trials} mismatches")


def build_solo_alphabet(alphabet: Alphabet) -> dict:
    source_name = SOLO_SOURCES[alphabet]
    source = _source(source_name)
    symbols = sorted(_filtered_symbols(source))
    first_symbols = sorted(_filtered_first_symbols(source))
    _verify_fusion(first_symbols, symbols, source)
    return {
        "name": alphabet.value,
        "source_encoding": source_name,
        "prefixes": PREFIXES,
        "first_symbols": first_symbols,
        "symbols": symbols,
    }


def build_shared_alphabet() -> dict:
    sources = [_source(name) for name in SHARED_SOURCES]
    symbols = sorted(set.intersection(*(_filtered_symbols(s) for s in sources)))
    first_symbols = sorted(set.intersection(*(_filtered_first_symbols(s) for s in sources)))
    for source in sources:
        _verify_fusion(first_symbols, symbols, source)
    return {
        "name": Alphabet.SHARED_V1.value,
        "source_encodings": SHARED_SOURCES,
        "prefixes": PREFIXES,
        "first_symbols": first_symbols,
        "symbols": symbols,
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for alphabet, source_name in SOLO_SOURCES.items():
        data = build_solo_alphabet(alphabet)
        out_path = DATA_DIR / f"{alphabet.value}.json"
        out_path.write_text(json.dumps(data, indent=0))
        print(f"{alphabet.value} (from {source_name}): "
              f"{len(data['first_symbols'])} first symbols, "
              f"{len(data['symbols'])} symbols -> {out_path}")

    data = build_shared_alphabet()
    out_path = DATA_DIR / f"{Alphabet.SHARED_V1.value}.json"
    out_path.write_text(json.dumps(data, indent=0))
    print(f"{Alphabet.SHARED_V1.value} (from {', '.join(SHARED_SOURCES)}): "
          f"{len(data['first_symbols'])} first symbols, "
          f"{len(data['symbols'])} symbols -> {out_path}")


if __name__ == "__main__":
    main()
