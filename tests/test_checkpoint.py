from pathlib import Path

import pytest
import torch


def _model():
    from ftrec.models.sasrec import SASRec, SASRecConfig

    return SASRec(SASRecConfig(num_items=12, hidden_size=4, num_blocks=1, num_heads=1, dropout=0, maxlen=3))


def test_checkpoint_round_trip_restores_model_and_metadata(tmp_path: Path) -> None:
    from ftrec.training.checkpoint import load_checkpoint, save_checkpoint

    model = _model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, metadata={"data_hash": "data-a", "config_hash": "cfg-a"})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)

    loaded = load_checkpoint(path, model, expected={"data_hash": "data-a"})

    assert loaded.metadata["config_hash"] == "cfg-a"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_checkpoint_rejects_wrong_data_fingerprint(tmp_path: Path) -> None:
    from ftrec.training.checkpoint import CheckpointMismatchError, load_checkpoint, save_checkpoint

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _model(), metadata={"data_hash": "data-a"})

    with pytest.raises(CheckpointMismatchError, match="data_hash"):
        load_checkpoint(path, _model(), expected={"data_hash": "data-b"})


def test_checkpoint_rejects_tampered_model_state(tmp_path: Path) -> None:
    from ftrec.training.checkpoint import CheckpointMismatchError, load_checkpoint, save_checkpoint

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _model(), metadata={"data_hash": "data-a"})
    payload = torch.load(path, weights_only=False)
    payload["model"]["final_norm.bias"][0] += 1
    torch.save(payload, path)

    with pytest.raises(CheckpointMismatchError, match="model_state_hash"):
        load_checkpoint(path, _model())
