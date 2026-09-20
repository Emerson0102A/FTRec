"""Export LLM2Attr embeddings with explicit FTRec item IDs."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from ftrec.artifacts import RunDirectory

from .artifacts import create_embedding_arrays, finish_artifact
from .checkpoints import resolve_checkpoint_paths
from .provider_utils import batched, iter_catalog, load_catalog_manifest, move_to_device


DEFAULT_SERVER_ROOT = Path("/root/autodl-tmp/FTRec/LLM2Attr")


def attribute_prompt(title: str, mask_token: str, count: int) -> str:
    instruction = (
        f"Input: \nGiven the title of an item, generate {count} concise attributes "
        "(only one word) that describe its key features from different perspectives."
    )
    slots = " ".join(f"Attribute {index + 1}: {mask_token}." for index in range(count))
    return f"{instruction}\nItem: {title}. \nOutput: \n{slots}"


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LLM2Attr export requires PyTorch") from exc
    checkpoints = resolve_checkpoint_paths(
        args.llm2attr_root,
        mntp_checkpoint=args.mntp_checkpoint,
        attribute_checkpoint=args.attribute_checkpoint,
    )
    sys.path.insert(0, str(checkpoints.root))
    try:
        from LLM2Attr import LLM2Attr
        from PLoRAModel import PLoRAModel
    except ImportError as exc:
        raise RuntimeError("install requirements-llm2attr.txt before export") from exc
    dtype = args.torch_dtype if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    model = LLM2Attr.from_pretrained(
        base_model_name_or_path=args.model,
        enable_bidirectional=True,
        peft_model_name_or_path=str(checkpoints.mntp),
        merge_peft=True,
        pooling_mode="mean",
        max_length=args.max_length,
        torch_dtype=dtype,
        attn_implementation=args.attention,
        attention_dropout=0.0,
    )
    tokenizer = model.tokenizer
    if tokenizer.mask_token is None:
        tokenizer.add_tokens(["<mask>"])
        tokenizer.mask_token = "<mask>"
        embeddings = model.model.get_input_embeddings()
        if len(tokenizer) > embeddings.num_embeddings:
            model.model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.model = PLoRAModel.from_pretrained(
        model.model, str(checkpoints.attribute), prompt_names=["title", "attr"]
    )
    model.to(args.device)
    model.eval()
    return model, tokenizer, torch, checkpoints


def export_llm2attr(args: argparse.Namespace) -> Path:
    model, tokenizer, torch, checkpoints = _load_model(args)
    catalog_manifest = load_catalog_manifest(args.catalog)
    item_count = int(catalog_manifest["item_count"])
    with RunDirectory(args.output, force=args.force) as run:
        assert run.path is not None
        title_array = attribute_array = present_array = None
        processed = 0
        with gzip.open(run.path / "attributes.jsonl.gz", "wt", encoding="utf-8") as stream:
            for rows in batched(iter_catalog(args.catalog), args.batch_size):
                if args.max_items is not None:
                    remaining = args.max_items - processed
                    if remaining <= 0:
                        break
                    rows = rows[:remaining]
                titles = [str(row.get("title") or row["parent_asin"]) for row in rows]
                prompts = [attribute_prompt(title, tokenizer.mask_token, args.attribute_count) for title in titles]
                title_features = move_to_device(model.tokenize(titles), args.device)
                attr_features = move_to_device(model.tokenize(prompts), args.device)
                with torch.inference_mode():
                    title_embeddings = model(title_features, prompt_name="title")
                    attr_embeddings, attr_ids = model.get_attr(
                        attr_features, prompt_name="attr", beam_size=0, print_attr=False
                    )
                title_values = title_embeddings.detach().float().cpu().numpy()
                attr_values = attr_embeddings.detach().float().cpu().numpy()
                ids = attr_ids.detach().cpu().numpy()
                if title_array is None:
                    title_array, attribute_array, present_array = create_embedding_arrays(
                        run.path, item_count=item_count,
                        attribute_count=args.attribute_count,
                        embedding_dim=int(title_values.shape[-1]), dtype=args.output_dtype,
                    )
                if attr_values.shape[1] != args.attribute_count:
                    raise RuntimeError(
                        f"model returned {attr_values.shape[1]} attributes; expected {args.attribute_count}"
                    )
                for offset, row in enumerate(rows):
                    item_id = int(row["item_id"])
                    title_array[item_id] = title_values[offset]
                    attribute_array[item_id] = attr_values[offset]
                    present_array[item_id] = bool(row.get("title"))
                    tokens = [tokenizer.decode([int(token)], skip_special_tokens=True).strip() for token in ids[offset]]
                    stream.write(json.dumps({
                        "item_id": item_id, "parent_asin": row["parent_asin"],
                        "attributes": tokens,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                processed += len(rows)
                print(f"LLM2Attr items: {processed}/{args.max_items or item_count}", flush=True)
        if title_array is None or attribute_array is None or present_array is None:
            raise RuntimeError("catalog contained no rows")
        finish_artifact(
            run.path, provider="llm2attr", item_count=item_count,
            attribute_count=args.attribute_count, embedding_dim=int(title_array.shape[-1]),
            catalog_sha256=str(catalog_manifest["catalog_sha256"]), title=title_array,
            attributes=attribute_array, present=present_array,
            provider_config={
                "model": args.model, "mntp_checkpoint": str(checkpoints.mntp),
                "attribute_checkpoint": str(checkpoints.attribute),
                "llm2attr_root": str(checkpoints.root), "processed_items": processed,
                "max_length": args.max_length, "torch_dtype": args.torch_dtype,
                "attention": args.attention, "batch_size": args.batch_size,
            },
        )
        run.complete({"provider": "llm2attr", "processed_items": processed, "total_items": item_count})
    return Path(args.output).resolve() / "artifact.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm2attr-root", type=Path, default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--mntp-checkpoint", type=Path)
    parser.add_argument("--attribute-checkpoint", type=Path)
    parser.add_argument("--attribute-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.attribute_count, args.max_length) < 1:
        parser.error("batch size, attribute count, and max length must be positive")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    return args


def main() -> None:
    print(export_llm2attr(parse_args()))


if __name__ == "__main__":
    main()
