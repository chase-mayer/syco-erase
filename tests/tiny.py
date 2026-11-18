"""A tiny random Qwen2 model with a character-level tokenizer, for offline tests.

    python -m tests.tiny /path/to/tiny_model

Every printable ASCII character is one token, so '(' and each answer letter are single
tokens exactly as they are for the real Qwen tokenizer. A minimal chat template is included
so the chat-format code path can be exercised too.
"""

from __future__ import annotations

import string
import sys

import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    chars = list(string.printable) + ["’", "“", "”", "–", "√", "°"]
    special = ["<unk>", "<pad>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    vocab = {t: i for i, t in enumerate(special + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  eos_token="<|end|>")
    tok.add_special_tokens({"additional_special_tokens": special[3:]})
    tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"
    return tok


def tiny_model(vocab_size: int, layers: int = 4, d: int = 32) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(vocab_size=vocab_size, hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def build(out_dir: str, layers: int = 4) -> str:
    tok = tiny_tokenizer()
    model = tiny_model(len(tok), layers=layers)
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "tiny_model"))
