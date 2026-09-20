"""Export schema-constrained LLM attributes using the shared artifact format.

Use ``--input-fields title`` for the controlled comparison with LLM2Attr.  The
enhanced experiment may additionally use catalog metadata, but it must be
reported separately because it receives more evidence.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ftrec.artifacts import RunDirectory

from .artifacts import create_embedding_arrays, finish_artifact
from .provider_utils import batched, iter_catalog, load_catalog_manifest, move_to_device


SYSTEM_PROMPT = (
    "You extract objective product attributes for recommendation. "
    "Return valid JSON only and never add explanations."
)


def _compact(value: Any, max_characters: int) -> Any:
    if isinstance(value, str):
        return value[:max_characters]
    if isinstance(value, list):
        result: list[str] = []
        used = 0
        for item in value:
            text = str(item)
            if used + len(text) > max_characters:
                break
            result.append(text)
            used += len(text)
        return result
    if isinstance(value, dict):
        result: dict[str, str] = {}
        used = 0
        for key, item in value.items():
            text = str(item)
            if used + len(str(key)) + len(text) > max_characters:
                break
            result[str(key)] = text
            used += len(str(key)) + len(text)
        return result
    return value


def build_prompt(
    row: dict[str, Any],
    *,
    attribute_count: int,
    input_fields: Iterable[str],
    max_field_characters: int,
) -> str:
    evidence = {
        field: _compact(row.get(field, ""), max_field_characters)
        for field in input_fields
        if field in row
    }
    schema = json.dumps(
        {"attributes": [f"attribute {index + 1}" for index in range(attribute_count)]},
        separators=(",", ":"),
    )
    return (
        f"Extract exactly {attribute_count} distinct, concise English product "
        "attributes. Use different semantic aspects when possible, avoid "
        "advertising language, and do not infer unsupported facts. Each "
        f"attribute must be at most four words. Return this schema exactly: {schema}.\n"
        f"Product evidence: {json.dumps(evidence, ensure_ascii=False, sort_keys=True)}"
    )


def _normalize_attribute(value: Any) -> str:
    return " ".join(str(value).strip().strip("-•.,;:\"'").split())[:80]


def parse_attributes(text: str, attribute_count: int) -> list[str]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE
    )
    candidates: list[Any] = []
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict) and isinstance(parsed.get("attributes"), list):
                candidates = parsed["attributes"]
        except json.JSONDecodeError:
            pass
    if not candidates:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, list):
                    candidates = parsed
            except json.JSONDecodeError:
                pass
    if not candidates:
        candidates = re.split(r"[,;\n]", cleaned)

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_attribute(candidate)
        key = normalized.casefold()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
        if len(result) == attribute_count:
            break
    return result + [""] * (attribute_count - len(result))


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "structured extraction requires PyTorch and transformers"
        ) from exc
    dtype = (
        args.torch_dtype
        if args.torch_dtype == "auto"
        else getattr(torch, args.torch_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attention,
    ).to(args.device)
    model.eval()
    return model, tokenizer, torch


def _chat_prompts(tokenizer: Any, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def _encode_texts(
    model: Any,
    tokenizer: Any,
    torch: Any,
    texts: list[str],
    args: argparse.Namespace,
) -> Any:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = move_to_device(encoded, args.device)
    outputs = model(
        **encoded, output_hidden_states=True, use_cache=False, return_dict=True
    )
    return _mean_pool(outputs.hidden_states[-1], encoded["attention_mask"])


def export_structured(args: argparse.Namespace) -> Path:
    model, tokenizer, torch = _load_model(args)
    catalog_manifest = load_catalog_manifest(args.catalog)
    item_count = int(catalog_manifest["item_count"])
    fields = tuple(part.strip() for part in args.input_fields.split(",") if part.strip())
    if "title" not in fields:
        raise ValueError("input-fields must include title")

    with RunDirectory(args.output, force=args.force) as run:
        assert run.path is not None
        title_array = attribute_array = present_array = None
        processed = 0
        with gzip.open(
            run.path / "attributes.jsonl.gz", "wt", encoding="utf-8"
        ) as extracted:
            for rows in batched(iter_catalog(args.catalog), args.batch_size):
                if args.max_items is not None:
                    remaining = args.max_items - processed
                    if remaining <= 0:
                        break
                    rows = rows[:remaining]
                prompts = [
                    build_prompt(
                        row,
                        attribute_count=args.attribute_count,
                        input_fields=fields,
                        max_field_characters=args.max_field_characters,
                    )
                    for row in rows
                ]
                encoded = tokenizer(
                    _chat_prompts(tokenizer, prompts),
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                encoded = move_to_device(encoded, args.device)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    continuation = generated_ids[:, encoded["input_ids"].shape[1] :]
                    generations = tokenizer.batch_decode(
                        continuation, skip_special_tokens=True
                    )
                    attributes = [
                        parse_attributes(text, args.attribute_count)
                        for text in generations
                    ]
                    titles = [
                        str(row.get("title") or row["parent_asin"]) for row in rows
                    ]
                    flat_texts = titles + [
                        attribute for group in attributes for attribute in group
                    ]
                    embeddings = _encode_texts(
                        model, tokenizer, torch, flat_texts, args
                    )

                values = embeddings.detach().float().cpu().numpy()
                row_count = len(rows)
                title_values = values[:row_count]
                attribute_values = values[row_count:].reshape(
                    row_count, args.attribute_count, -1
                )
                for row_index, group in enumerate(attributes):
                    for attribute_index, attribute in enumerate(group):
                        if not attribute:
                            attribute_values[row_index, attribute_index] = 0
                if title_array is None:
                    title_array, attribute_array, present_array = (
                        create_embedding_arrays(
                            run.path,
                            item_count=item_count,
                            attribute_count=args.attribute_count,
                            embedding_dim=int(title_values.shape[-1]),
                            dtype=args.output_dtype,
                        )
                    )
                for offset, row in enumerate(rows):
                    item_id = int(row["item_id"])
                    title_array[item_id] = title_values[offset]
                    attribute_array[item_id] = attribute_values[offset]
                    present_array[item_id] = bool(row.get("title"))
                    extracted.write(
                        json.dumps(
                            {
                                "item_id": item_id,
                                "parent_asin": row["parent_asin"],
                                "attributes": attributes[offset],
                                "raw_generation": generations[offset],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                processed += len(rows)
                print(
                    f"Structured-LLM items: {processed}/{args.max_items or item_count}",
                    flush=True,
                )
        if title_array is None or attribute_array is None or present_array is None:
            raise RuntimeError("catalog contained no rows")
        finish_artifact(
            run.path,
            provider="structured-llm",
            item_count=item_count,
            attribute_count=args.attribute_count,
            embedding_dim=int(title_array.shape[-1]),
            catalog_sha256=str(catalog_manifest["catalog_sha256"]),
            title=title_array,
            attributes=attribute_array,
            present=present_array,
            provider_config={
                "model": args.model,
                "input_fields": list(fields),
                "processed_items": processed,
                "max_length": args.max_length,
                "max_new_tokens": args.max_new_tokens,
                "max_field_characters": args.max_field_characters,
                "decoding": "greedy",
                "torch_dtype": args.torch_dtype,
                "attention": args.attention,
                "batch_size": args.batch_size,
            },
        )
        run.complete(
            {
                "provider": "structured-llm",
                "input_fields": list(fields),
                "processed_items": processed,
                "total_items": item_count,
            }
        )
    return Path(args.output).resolve() / "artifact.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--input-fields", default="title")
    parser.add_argument("--attribute-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-field-characters", type=int, default=1200)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--attention",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.attribute_count,
        args.max_length,
        args.max_new_tokens,
        args.max_field_characters,
    ) < 1:
        parser.error("batch sizes, counts, lengths, and limits must be positive")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    return args


def main() -> None:
    print(export_structured(parse_args()))


if __name__ == "__main__":
    main()
