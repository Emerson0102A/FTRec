"""Locate and validate the released LLM2Attr adapter checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MNTP_RELATIVE_PATH = Path("output/mntp/Qwen2.5-0.5B/checkpoint-10000")
ATTRIBUTE_RELATIVE_PATH = Path("output/attr/Qwen2.5-0.5B/checkpoint-3000")


@dataclass(frozen=True)
class CheckpointPaths:
    root: Path
    mntp: Path
    attribute: Path


def _require_files(root: Path, relative_files: tuple[str, ...], label: str) -> None:
    missing = [name for name in relative_files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label} checkpoint is incomplete at {root}: {', '.join(missing)}"
        )


def resolve_checkpoint_paths(
    llm2attr_root: str | Path,
    *,
    mntp_checkpoint: str | Path | None = None,
    attribute_checkpoint: str | Path | None = None,
) -> CheckpointPaths:
    root = Path(llm2attr_root).expanduser().resolve()
    if not (root / "LLM2Attr.py").is_file() or not (root / "PLoRAModel.py").is_file():
        raise FileNotFoundError(
            f"LLM2Attr source directory is incomplete: {root} "
            "(expected LLM2Attr.py and PLoRAModel.py)"
        )
    mntp = (
        Path(mntp_checkpoint).expanduser().resolve()
        if mntp_checkpoint is not None
        else root / MNTP_RELATIVE_PATH
    )
    attribute = (
        Path(attribute_checkpoint).expanduser().resolve()
        if attribute_checkpoint is not None
        else root / ATTRIBUTE_RELATIVE_PATH
    )
    _require_files(mntp, ("adapter_config.json", "adapter_model.safetensors"), "MNTP")
    _require_files(
        attribute,
        (
            "adapter_config.json",
            "adapter_model.safetensors",
            "title_prompt_encoder/adapter_config.json",
            "title_prompt_encoder/prompt_tokens.pt",
            "title_prompt_encoder/pytorch_model.bin",
            "attr_prompt_encoder/adapter_config.json",
            "attr_prompt_encoder/prompt_tokens.pt",
            "attr_prompt_encoder/pytorch_model.bin",
        ),
        "attribute",
    )
    return CheckpointPaths(root=root, mntp=mntp, attribute=attribute)
