#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B inference using the retained best-round prefill/decode paths.

Copy this directory as a whole. See README.md for dependencies and examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from solution import entry
from solution.qwen38_optimized import HybridCache, Qwen38 as _BaseModel


class Qwen38(_BaseModel):
    """Single-GPU text model with optimized forward and greedy generation."""

    def forward_step(self, input_ids, kv_cache, *, request_indices=None, return_all_logits=False):
        return entry.run(self, input_ids, kv_cache, request_indices=request_indices,
                         return_all_logits=return_all_logits)

    def generate(self, input_ids, *, max_new_tokens=None, eos_token_ids=None, prefill_chunk_size=512):
        return entry.generate(self, input_ids, max_new_tokens=max_new_tokens,
                              eos_token_ids=eos_token_ids, prefill_chunk_size=prefill_chunk_size)


run = entry.run
forward_step = run
generate = entry.generate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Local Qwen3.8-27B checkpoint directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", type=int, default=8192)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-ids", help="JSON flat token list or batch of token lists")
    source.add_argument("--prompt", action="append", help="Repeat for a batch; uses the checkpoint chat template")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    tokenizer = None
    if args.input_ids is not None:
        ids = json.loads(args.input_ids)
    else:
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        path = Path(args.model_path)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        def raise_exception(message):
            raise ValueError(message)
        env.globals["raise_exception"] = raise_exception
        template = env.from_string((path / "chat_template.jinja").read_text())
        ids = [tokenizer.encode(template.render(messages=[{"role": "user", "content": prompt}],
                add_generation_prompt=True, enable_thinking=args.enable_thinking),
                add_special_tokens=False).ids for prompt in args.prompt]
    model = Qwen38(args.model_path, args.device, max_context=args.max_context)
    result = model.generate(ids, max_new_tokens=args.max_new_tokens, prefill_chunk_size=args.prefill_chunk_size)
    print(json.dumps({"output_ids": result,
                      "text": [tokenizer.decode(row, skip_special_tokens=False) for row in result] if tokenizer else None},
                     ensure_ascii=False))


__all__ = ["Qwen38", "HybridCache", "forward_step", "run", "generate"]

if __name__ == "__main__":
    main()
