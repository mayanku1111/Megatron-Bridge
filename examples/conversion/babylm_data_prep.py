#!/usr/bin/env python3
"""
Usage
    # Small test slice (~100MB output):
    python babylm_data_prep.py \
        --output-prefix /data/babylm_test \
        --hf-tokenizer bharatgenai/Param2-17B-A2.4B-Thinking \
        --workers 4 \
        --max-rows 170000

    # Tiny smoke-test (just verify pipeline works, ~6MB):
    python babylm_data_prep.py \
        --output-prefix /data/babylm_smoke \
        --hf-tokenizer bharatgenai/Param2-17B-A2.4B-Thinking \
        --workers 2 \
        --max-rows 10000

Output
------
    <output-prefix>_text_document.bin
    <output-prefix>_text_document.idx

Then pass to training:
    'dataset.blend=[[<output-prefix>_text_document],null]'

Size reference (BabyLM avg ~148 tokens/row):
    --max-rows 10_000  →  ~6  MB   (smoke test only)
    --max-rows 50_000  →  ~30 MB
    --max-rows 170_000 →  ~100 MB  (recommended for testing)
    --max-rows 500_000 →  ~300 MB
    (no limit)         →  ~640 MB  (full 11.1M rows)

Batch size guide (Param2-17B, seq_length=4096, with activation recompute):
    1  × A100-80G  EP=1  → global_batch_size=2,  micro_batch_size=1
    8  × A100-80G  EP=8  → global_batch_size=64, micro_batch_size=1
    8  × H100-80G  EP=8  → global_batch_size=64, micro_batch_size=2
"""

import argparse
import multiprocessing as mp
import os
import struct

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# ── Megatron binary format constants ─────────────────────────────────────────
MEGATRON_MAGIC   = b"MMIDIDX\x00\x00"
DTYPE_CODE_INT32 = 4   # int32 covers vocab_size=128008 (uint16 only goes to 65535)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--output-prefix",
        default="./data/babylm",
        help="Output file prefix. Creates <prefix>_text_document.{bin,idx}",
    )
    p.add_argument(
        "--hf-tokenizer",
        default="bharatgenai/Param2-17B-A2.4B-Thinking",
        help="HuggingFace tokenizer ID or local path",
    )
    p.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Max sequence length. Documents longer than this are chunked. "
             "Must match model.seq_length in training config (default: 4096)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Stop after this many dataset rows (for testing). "
            "~170000 rows ≈ 100MB. Leave unset for full dataset."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of tokenisation worker processes",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Rows per worker chunk (pool.imap chunksize)",
    )
    p.add_argument(
        "--append-eod",
        action="store_true",
        default=True,
        help="Append EOS token to every document (default: True)",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=50_000,
        help="Print progress every N rows",
    )
    p.add_argument(
        "--valid-split",
        type=float,
        default=0.005,
        help=(
            "Fraction of data to reserve for validation (default: 0.005 = 0.5%%). "
            "Set to 0 to skip. Used to print the Megatron split string."
        ),
    )
    return p.parse_args()


# ── Tokeniser worker ──────────────────────────────────────────────────────────

_tok = None  # module-level singleton, one per worker process

def _init_worker(hf_tokenizer_path: str):
    global _tok
    _tok = AutoTokenizer.from_pretrained(
        hf_tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
    )

def _tokenize(args_tuple):
    """Tokenise one row. Returns list[int] of token ids."""
    text, append_eod = args_tuple
    if not text or not text.strip():
        return []
    ids = _tok.encode(text, add_special_tokens=False)
    if append_eod:
        eod = _tok.eos_token_id if _tok.eos_token_id is not None else 3
        ids = ids + [eod]
    return ids


# ── Megatron binary writer ────────────────────────────────────────────────────

class MegatronBinaryWriter:
    """
    Writes Megatron indexed dataset v1 (.bin + .idx).
    Compatible with megatron.core.datasets.indexed_dataset.
    Uses int32 to support vocab_size=128008.
    """

    def __init__(self, prefix: str):
        out_dir = os.path.dirname(os.path.abspath(prefix))
        os.makedirs(out_dir, exist_ok=True)
        self.bin_path  = f"{prefix}_text_document.bin"
        self.idx_path  = f"{prefix}_text_document.idx"
        self._bin      = open(self.bin_path, "wb")
        self._sizes    = []    # document lengths in tokens
        self._pointers = []    # byte offsets in .bin
        self._offset   = 0

    def add(self, tokens: list):
        arr = np.array(tokens, dtype=np.int32)
        self._bin.write(arr.tobytes())
        self._pointers.append(self._offset)
        self._sizes.append(len(tokens))
        self._offset += arr.nbytes

    def finalize(self):
        self._bin.close()
        n = len(self._sizes)
        with open(self.idx_path, "wb") as f:
            f.write(MEGATRON_MAGIC)
            f.write(struct.pack("<B", 1))                # version
            f.write(struct.pack("<B", DTYPE_CODE_INT32)) # dtype
            f.write(struct.pack("<Q", n))                # num documents
            f.write(struct.pack("<Q", n))                # num documents (repeated)
            f.write(np.array(self._sizes,    dtype=np.int32).tobytes())
            f.write(np.array(self._pointers, dtype=np.int64).tobytes())

        size_mb = self._offset / 1024 / 1024
        print(f"\n{'─'*50}")
        print(f"Output files:")
        print(f"  {self.bin_path}")
        print(f"  {self.idx_path}")
        print(f"  Documents : {n:,}")
        print(f"  Tokens    : {sum(self._sizes):,}")
        print(f"  Size      : {size_mb:.1f} MB")
        print(f"{'─'*50}")
        return n, sum(self._sizes)


# ── Main ──────────────────────────────────────────────────────────────────────

def print_batch_size_guide(total_tokens: int, seq_length: int):
    """Print recommended batch sizes based on dataset size."""
    total_seqs = total_tokens // seq_length
    print(f"\nBatch size guide (seq_length={seq_length}):")
    print(f"  Total sequences in dataset : ~{total_seqs:,}")
    print()
    configs = [
        ("1×A100-80G  EP=1 (single GPU)", 2,  1),
        ("8×A100-80G  EP=8 (1 node)    ", 64, 1),
        ("8×H100-80G  EP=8 (1 node)    ", 64, 2),
        ("32×H100-80G EP=16 (4 nodes)  ", 256, 1),
    ]
    for label, gbs, mbs in configs:
        iters = total_seqs // gbs
        print(f"  {label}  global_batch_size={gbs:<4}  micro_batch_size={mbs}"
              f"  → ~{iters:,} iters to see all data once")
    print()


def main():
    args = parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    limit_str = f" (first {args.max_rows:,} rows)" if args.max_rows else " (full dataset)"
    print(f"Loading BabyLM-2026-Strict{limit_str} …")

    ds = load_dataset(
        "BabyLM-community/BabyLM-2026-Strict",
        split="train",
        trust_remote_code=True,
    )
    total_available = len(ds)
    rows_to_process = min(args.max_rows, total_available) if args.max_rows else total_available
    print(f"  Available : {total_available:,} rows")
    print(f"  Processing: {rows_to_process:,} rows")

    # ── Tokenise ─────────────────────────────────────────────────────────────
    writer = MegatronBinaryWriter(args.output_prefix)

    pool = mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.hf_tokenizer,),
    )

    def _row_iter():
        for i, row in enumerate(ds):
            if args.max_rows and i >= args.max_rows:
                break
            yield (row["text"], args.append_eod)

    total_docs = 0
    total_toks = 0
    skipped    = 0

    print(f"\nTokenising with {args.workers} workers …")
    for i, tokens in enumerate(
        pool.imap(_tokenize, _row_iter(), chunksize=args.chunk_size)
    ):
        if not tokens:
            skipped += 1
            continue

        # Chunk documents that exceed seq_length
        for start in range(0, len(tokens), args.seq_length):
            chunk = tokens[start : start + args.seq_length]
            writer.add(chunk)
            total_toks += len(chunk)

        total_docs += 1

        if (i + 1) % args.log_interval == 0:
            pct = (i + 1) / rows_to_process * 100
            print(f"  [{i+1:>8,} / {rows_to_process:,}  {pct:5.1f}%]"
                  f"  docs={total_docs:,}  tokens={total_toks:,}")

    pool.close()
    pool.join()

    # ── Write index ───────────────────────────────────────────────────────────
    n_docs, n_toks = writer.finalize()

    if skipped:
        print(f"  Skipped {skipped:,} empty rows")

    # ── Print next steps ──────────────────────────────────────────────────────
    print_batch_size_guide(n_toks, args.seq_length)

    # Megatron split string: 99.5% train / 0.5% valid (if valid_split > 0)
    if args.valid_split > 0:
        train_pct = round((1 - args.valid_split) * 1000) / 10
        valid_pct = round(args.valid_split * 1000) / 10
        split_str = f"{train_pct},{valid_pct},0"
    else:
        split_str = "100,0,0"

    prefix = args.output_prefix
    print("Next steps:")
    print()
    print("  # Smoke test (1 GPU):")
    print(f"  torchrun --nproc-per-node=1 scripts/training/run_recipe.py \\")
    print(f"      --recipe param2_17b_pretrain_config \\")
    print(f"      --dataset llm-pretrain \\")
    print(f"      'dataset.blend=[[{prefix}_text_document],null]' \\")
    print(f"      dataset.split='{split_str}' \\")
    print(f"      model.expert_model_parallel_size=1 \\")
    print(f"      train.global_batch_size=2 \\")
    print(f"      train.micro_batch_size=1 \\")
    print(f"      train.train_iters=100")
    print()
    print("  # 8-GPU training:")
    print(f"  torchrun --nproc-per-node=8 scripts/training/run_recipe.py \\")
    print(f"      --recipe param2_17b_pretrain_config \\")
    print(f"      --dataset llm-pretrain \\")
    print(f"      'dataset.blend=[[{prefix}_text_document],null]' \\")
    print(f"      dataset.split='{split_str}' \\")
    print(f"      train.global_batch_size=64 \\")
    print(f"      train.micro_batch_size=1")
    print()


if __name__ == "__main__":
    main()