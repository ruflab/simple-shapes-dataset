import torch
import torch.utils.data
from utils import PROJECT_DIR

from simple_shapes_dataset.dataset import SimpleShapesDataset
from simple_shapes_dataset.domain import get_default_domains_dataset
from simple_shapes_dataset.pre_process import attribute_to_tensor


def test_attr_preprocess():
    transform = {
        "attr": attribute_to_tensor,
    }
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        domain_classes=get_default_domains_dataset(
            ["attr"],
            PROJECT_DIR / "sample_dataset",
            split="train",
            transforms=transform,
        ),
        transforms=transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    assert isinstance(item["attr"], list)
    assert len(item["attr"]) == 2
    assert isinstance(item["attr"][0], torch.Tensor)
    assert item["attr"][0].shape == (2, 3)
    assert isinstance(item["attr"][1], torch.Tensor)
    assert item["attr"][1].shape == (2, 8)
