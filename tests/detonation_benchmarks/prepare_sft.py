"""Tokenize a chat SFT dataset into a flat token file plus a loss mask.

Each conversation is rendered with the chat template of --template-model and
concatenated into <out>.bin (uint32 tokens); <out>.mask (uint8) marks the
assistant tokens, the only ones that contribute to the loss. Conversations are
shuffled with a fixed seed; the last --val-examples of them are written at
the tail, and their token count is stored in <out>.json as val_tokens, for
train_lm.py --val-tokens.

    python prepare_sft.py --out /scratch/.../tulu3_olmo2 --examples 200000
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="allenai/tulu-3-sft-olmo-2-mixture-0225")
    p.add_argument("--template-model", default="allenai/OLMo-2-0425-1B-SFT")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--examples", type=int, default=200_000, help="training conversations")
    p.add_argument("--val-examples", type=int, default=2_000)
    p.add_argument("--max-tokens", type=int, default=4096, help="skip longer conversations")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.template_model)
    ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    wanted = args.examples + args.val_examples

    def render(messages, **kwargs):
        return tok.apply_chat_template(messages, tokenize=True, **kwargs)

    tokens, masks, val_start, kept, skipped = [], [], None, 0, 0
    for example in ds:
        if kept == wanted:
            break
        messages = example["messages"]
        if not messages or messages[0]["role"] == "assistant":
            skipped += 1  # No prompt to condition the reply on.
            continue
        ids = render(messages)
        if len(ids) > args.max_tokens:
            skipped += 1
            continue
        mask = np.zeros(len(ids), dtype=np.uint8)
        consistent = True
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            start = len(render(messages[:i], add_generation_prompt=True))
            end = len(render(messages[: i + 1]))
            if ids[:end] != render(messages[: i + 1]):
                consistent = False
                break
            mask[start:end] = 1
        if not consistent or not mask.any():
            skipped += 1
            continue
        if kept == args.examples:
            val_start = sum(len(t) for t in tokens)
        tokens.append(np.asarray(ids, dtype=np.uint32))
        masks.append(mask)
        kept += 1

    flat_tokens = np.concatenate(tokens)
    flat_mask = np.concatenate(masks)
    flat_tokens.tofile(args.out + ".bin")
    flat_mask.tofile(args.out + ".mask")
    meta = {
        "dataset": args.dataset,
        "template_model": args.template_model,
        "train_examples": args.examples,
        "val_examples": args.val_examples,
        "skipped": skipped,
        "tokens": int(flat_tokens.size),
        "loss_tokens": int(flat_mask.sum()),
        "val_tokens": int(flat_tokens.size - val_start),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
