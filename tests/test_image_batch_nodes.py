import pytest
import torch

from ..helpers.image_helpers import split_image_batch_into_batches
from ..nodes.image_nodes import UC_ImageBatchToList, UC_ImageBatchToBatchList


def test_split_image_batch_into_batches_edge_cases():
    assert split_image_batch_into_batches(None, 8) == []
    assert split_image_batch_into_batches(torch.zeros((0, 32, 32, 3)), 8) == []

    # 3D tensor gets unsqueezed
    single_3d = torch.zeros((32, 32, 3))
    res_3d = split_image_batch_into_batches(single_3d, 8)
    assert len(res_3d) == 1
    assert res_3d[0].shape == (1, 32, 32, 3)


def test_split_image_batch_into_batches_220_by_8():
    # Exactly matching user's scenario: 220 images, batch_size 8 -> 28 batches (27 of size 8, 1 of size 4)
    images = torch.arange(220, dtype=torch.float32).view(220, 1, 1, 1).repeat(1, 16, 16, 3)
    batches = split_image_batch_into_batches(images, 8)
    assert len(batches) == 28

    for i in range(27):
        assert batches[i].shape == (8, 16, 16, 3)
        assert batches[i][0, 0, 0, 0].item() == float(i * 8)
        assert batches[i][-1, 0, 0, 0].item() == float(i * 8 + 7)

    assert batches[27].shape == (4, 16, 16, 3)
    assert batches[27][0, 0, 0, 0].item() == 216.0
    assert batches[27][-1, 0, 0, 0].item() == 219.0


def test_image_batch_to_batch_list_node_schema_and_execution():
    schema = UC_ImageBatchToBatchList.define_schema()
    assert schema.node_id == "UC_ImageBatchToBatchList"
    assert schema.display_name == "Image Batch to Batch List"
    assert schema.category == "utils"
    assert len(schema.inputs) == 2
    assert schema.inputs[0].id == "images"
    assert schema.inputs[1].id == "batch_size"
    assert schema.inputs[1].default == 8
    assert len(schema.outputs) == 1
    assert schema.outputs[0].is_output_list is True

    # Execute node
    images = torch.zeros((220, 32, 32, 3))
    output = UC_ImageBatchToBatchList.execute(images, batch_size=8)
    assert isinstance(output, tuple)
    assert len(output) == 1
    batch_list = output[0]
    assert isinstance(batch_list, list)
    assert len(batch_list) == 28
    assert all(b.shape == (8, 32, 32, 3) for b in batch_list[:27])
    assert batch_list[27].shape == (4, 32, 32, 3)


def test_image_batch_to_list_node_execution():
    images = torch.zeros((5, 32, 32, 3))
    output = UC_ImageBatchToList.execute(images)
    assert isinstance(output, tuple)
    assert len(output) == 1
    item_list = output[0]
    assert len(item_list) == 5
    assert all(item.shape == (1, 32, 32, 3) for item in item_list)
