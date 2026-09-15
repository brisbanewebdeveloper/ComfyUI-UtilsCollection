import pathlib
import sys
import types

import pytest
import torch


PACKAGE_NAME = "utils_collection_lama_test"
PACKAGE_ROOT = pathlib.Path(__file__).parents[1]
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_lama_test.helpers import lama_helpers
from utils_collection_lama_test.models.lama import FFCResNetGenerator


def test_generator_matches_big_lama_state_contract():
    import comfy.ops

    model = FFCResNetGenerator(comfy.ops.disable_weight_init)
    state = model.state_dict()

    assert len(state) == 989
    assert state["model.1.ffc.convl2l.weight"].shape == (64, 4, 7, 7)
    assert state["model.4.ffc.convl2g.weight"].shape == (384, 256, 3, 3)
    assert state["model.34.weight"].shape == (3, 64, 7, 7)


def test_prepare_inputs_broadcasts_pads_and_preserves_standard_mask_semantics():
    images = torch.rand(2, 9, 10, 3)
    masks = torch.zeros(1, 9, 10)
    masks[:, 2:5, 3:7] = 1.0

    padded_images, binary_mask, original_size = lama_helpers.prepare_lama_inputs(
        images, masks, mask_threshold=250, gaussblur_radius=0, invert_mask=False
    )

    assert padded_images.shape == (2, 3, 16, 16)
    assert binary_mask.shape == (2, 1, 16, 16)
    assert original_size == (9, 10)
    assert torch.equal(binary_mask[0], binary_mask[1])
    assert binary_mask[0, 0, 3, 4] == 1
    assert binary_mask[0, 0, 0, 0] == 0
    assert binary_mask[0, 0, 15, 15] == 0


def test_prepare_inputs_inverts_mask():
    images = torch.zeros(1, 8, 8, 3)
    masks = torch.zeros(1, 8, 8)
    masks[:, 2:6, 2:6] = 1.0

    _, binary_mask, _ = lama_helpers.prepare_lama_inputs(
        images, masks, mask_threshold=250, gaussblur_radius=0, invert_mask=True
    )

    assert binary_mask[0, 0, 3, 3] == 0
    assert binary_mask[0, 0, 0, 0] == 1


def test_prepare_inputs_rejects_non_broadcastable_mask_batch():
    with pytest.raises(ValueError, match="mask batch"):
        lama_helpers.prepare_lama_inputs(
            torch.zeros(3, 8, 8, 3),
            torch.zeros(2, 8, 8),
            mask_threshold=250,
            gaussblur_radius=0,
            invert_mask=False,
        )


def test_image_mask_to_luma_uses_rgb_luminance():
    masks = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])

    result = lama_helpers.image_mask_to_luma(masks)

    assert torch.allclose(result, torch.tensor([[[0.299, 0.587]]]))


def test_lama_device_options_include_every_visible_gpu(monkeypatch):
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    monkeypatch.setattr(
        lama_helpers.model_management, "get_all_torch_devices", lambda: devices
    )

    assert lama_helpers.get_lama_device_options() == [
        "default",
        "cpu",
        "gpu:0",
        "gpu:1",
    ]
    assert lama_helpers.resolve_lama_device("gpu:1") == torch.device("cuda:1")


def test_lama_device_options_do_not_alias_cpu_as_gpu(monkeypatch):
    monkeypatch.setattr(
        lama_helpers.model_management,
        "get_all_torch_devices",
        lambda: [torch.device("cpu")],
    )

    assert lama_helpers.get_lama_device_options() == ["default", "cpu"]


def test_loader_streams_copies_releases_and_closes(monkeypatch, tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 2))
            self.register_buffer("counter", torch.empty((), dtype=torch.int64))

        def get_dtype(self):
            return self.weight.dtype

    class FakeStream:
        def __init__(self, batches):
            self.batches = iter(batches)
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.batches)

        def close(self):
            self.closed = True

    class FakeHandler:
        def __init__(self):
            self.tensors = {
                "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
                "counter": torch.tensor(9, dtype=torch.int64),
            }
            self.released = []
            self.closed = False
            self.stream = None
            self.stream_kwargs = None

        def keys(self):
            return self.tensors.keys()

        def metadata(self):
            return {"architecture": "FFCResNetGenerator"}

        def get_shape(self, key):
            return self.tensors[key].shape

        def get_dtype(self, key):
            return self.tensors[key].dtype

        def async_stream(self, keys, **kwargs):
            self.stream_kwargs = kwargs
            self.stream = FakeStream([[(key, self.tensors[key])] for key in keys])
            return self.stream

        def mark_processed(self, key):
            self.released.append(key)

        def close(self):
            self.closed = True

    source = tmp_path / "model.safetensors"
    source.write_bytes(b"fixture")
    handler = FakeHandler()
    monkeypatch.setattr(lama_helpers, "FFCResNetGenerator", lambda operations: TinyModel())
    monkeypatch.setattr(lama_helpers, "MemoryEfficientSafeOpen", lambda *args, **kwargs: handler)
    monkeypatch.setattr(
        lama_helpers.comfy.model_patcher,
        "CoreModelPatcher",
        lambda model, load_device, offload_device: types.SimpleNamespace(
            model=model, load_device=load_device, offload_device=offload_device
        ),
    )

    loaded = lama_helpers.load_lama_model(source, "cpu")

    assert torch.equal(loaded.model.weight, handler.tensors["weight"])
    assert loaded.model.counter.item() == 9
    assert handler.released == ["weight", "counter"]
    assert handler.stream_kwargs == {
        "batch_size": 1,
        "prefetch_batches": 1,
        "pin_memory": False,
    }
    assert handler.stream.closed
    assert handler.closed
