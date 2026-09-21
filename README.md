# llm-ids

Encode data as token-efficient, LLM-friendly strings.

Hex, base64, and UUID strings are the usual way to represent IDs and secrets, but generic tokenizers compress them poorly, costing many tokens per ID: 3.3-9.7 bits/token depending on tokenizer and the ID charset. llm-ids uses a bijective numeration over an alphabet of whole vocabulary tokens instead: **11.6-13.7 bits per token**.

## Install

```
pip install llm-ids
```

## Usage

```python
import hashlib, string
from llm_ids import Alphabet, Codec

int_codec = Codec(Alphabet.O200K_V1, charset=string.digits)
token_id = int_codec.to_token_id(str(42))                              # 'ACLE'
assert int_codec.from_token_id(token_id) == "42"

alnum_codec = Codec(Alphabet.O200K_V1, charset=string.ascii_uppercase + string.digits)
token_id = alnum_codec.to_token_id("A1B2C3")                           # 'Stocks.bunifu'
assert alnum_codec.from_token_id(token_id) == "A1B2C3"

hex_codec = Codec(Alphabet.O200K_V1, charset=string.digits + "abcdef")
digest_hex = hashlib.sha256(b"hello").hexdigest()
token_id = hex_codec.to_token_id(digest_hex)                           # 'Tournament_you...'
assert hex_codec.from_token_id(token_id) == digest_hex
```

Available alphabets (see `src/llm_ids/alphabets.py`):

| `Alphabet` member | use for | bits/token |
|---|---|---|
| `SHARED_V1` | OpenAI (o200k/cl100k), Llama 3.x, and Qwen3 at once | >=13.25 |
| `O200K_V1` | GPT-4o, GPT-4.1, GPT-5 family | >=13.36 |
| `CL100K_V1` | GPT-4, GPT-3.5-turbo | >=13.67 |
| `MISTRAL_TEKKEN_V1` | Mistral NeMo, Small/Large 2.x+ | >=12.14 |
| `DEEPSEEK_V1` | DeepSeek V3, R1 | >=11.58 |

Use a solo alphabet when the target tokenizer is known. Use `SHARED_V1` when it isn't, or spans several of those families, since it costs almost nothing (>=13.25 vs. >=13.36-13.67 bits/token). Mistral and DeepSeek get their own alphabets since their vocabularies barely overlap with the others.

> NOTE: Gemma is not supported. SentencePiece has no fixed pretoken boundary, so token counts are not provably fixed the way they are for BPE tokenizers.

## How it works

- `llm_ids.id_codec`: integer <-> token_id string, using the token-boundary trick above.
- `llm_ids.bijective`: generic bijective numeration; preserves leading "zero" symbols that plain positional notation would drop.
- `llm_ids.char_token_id`: strings over a caller-declared charset, built on `bijective` + `id_codec`.
- `llm_ids.codec.Codec`: the alphabet-and-charset-bound entry point above; use this.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```
uv sync --group dev
uv run pytest
```

To regenerate an alphabet's data file from its source tokenizer vocabulary
(requires the `tiktoken`/`tokenizers`/`huggingface_hub` dev dependencies):

```
uv run python scripts/build_alphabets.py
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix:`, `feat:`, `feat!:`/`BREAKING CHANGE:`), checked on every PR. A maintainer triggers a release manually from the Actions tab (`Release` workflow), which runs [Commitizen](https://commitizen-tools.github.io/commitizen/) to bump the version, update `CHANGELOG.md`, tag, and create a GitHub Release -- publishing to PyPI happens automatically from there.
