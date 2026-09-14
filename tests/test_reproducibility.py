import numpy as np
import torch


def test_seed_everything_repeats_torch_and_numpy() -> None:
    from ftrec.reproducibility import seed_everything

    seed_everything(42, deterministic=True)
    torch_first = torch.rand(4)
    numpy_first = np.random.rand(4)

    seed_everything(42, deterministic=True)

    assert torch.equal(torch_first, torch.rand(4))
    assert np.array_equal(numpy_first, np.random.rand(4))


def test_resolve_device_rejects_unavailable_cuda() -> None:
    from ftrec.reproducibility import DeviceError, resolve_device

    if not torch.cuda.is_available():
        try:
            resolve_device("cuda")
        except DeviceError as error:
            assert "CUDA" in str(error)
        else:
            raise AssertionError("unavailable CUDA should be rejected")

