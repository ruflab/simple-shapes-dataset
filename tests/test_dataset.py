from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor
from utils import PROJECT_DIR

from simple_shapes_dataset.data_module import SimpleShapesDataModule
from simple_shapes_dataset.dataset import SimpleShapesDataset
from simple_shapes_dataset.domain import (
    get_default_domains,
    get_default_domains_dataset,
)
from simple_shapes_dataset.domain_alignment import get_aligned_datasets
from simple_shapes_dataset.pre_process import attribute_to_tensor


def test_dataset():
    transform = {"attr": Compose([]), "v": Compose([])}

    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        domain_classes=get_default_domains_dataset(
            ["v", "attr"],
            PROJECT_DIR / "sample_dataset",
            split="train",
            transforms=transform,
        ),
    )

    assert len(dataset) == 4

    item = dataset[0]
    for domain in ["v", "attr"]:
        assert domain in item


def test_dataset_val():
    transform = {"attr": Compose([]), "v": Compose([])}
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="val",
        domain_classes=get_default_domains_dataset(
            ["v", "attr"],
            PROJECT_DIR / "sample_dataset",
            split="val",
            transforms=transform,
        ),
    )

    assert len(dataset) == 2

    item = dataset[0]
    for domain in ["v", "attr"]:
        assert domain in item


def test_dataloader():
    transform = {"v": ToTensor(), "attr": attribute_to_tensor}
    dataset = SimpleShapesDataset(
        PROJECT_DIR / "sample_dataset",
        split="train",
        domain_classes=get_default_domains_dataset(
            ["v", "attr"],
            PROJECT_DIR / "sample_dataset",
            split="train",
            transforms=transform,
        ),
        transforms=transform,
    )

    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))
    for domain in ["v", "attr"]:
        assert domain in item


def test_get_aligned_datasets():
    transform = {"t": Compose([]), "v": Compose([])}

    datasets = get_aligned_datasets(
        PROJECT_DIR / "sample_dataset",
        "train",
        domain_classes=get_default_domains_dataset(
            ["v", "t"],
            PROJECT_DIR / "sample_dataset",
            split="train",
            transforms=transform,
        ),
        domain_proportions={
            frozenset(["v", "t"]): 0.5,
            frozenset("v"): 1.0,
            frozenset("t"): 1.0,
        },
        max_size=4,
        seed=0,
    )

    assert len(datasets) == 3
    for dataset_name, _ in datasets.items():
        assert dataset_name in [
            frozenset(["v", "t"]),
            frozenset(["v"]),
            frozenset(["t"]),
        ]


def test_datamodule():
    datamodule = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        get_default_domains(["attr"]),
        domain_proportions={
            frozenset(["attr"]): 1.0,
        },
        batch_size=2,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    item, _, _ = next(iter(train_dataloader))
    assert isinstance(item, dict)
    assert len(item) == 1
    assert frozenset(["attr"]) in item


def test_datamodule_aligned_dataset():
    datamodule = SimpleShapesDataModule(
        PROJECT_DIR / "sample_dataset",
        get_default_domains(["v", "attr"]),
        domain_proportions={
            frozenset(["v", "attr"]): 0.5,
            frozenset(["v"]): 1.0,
            frozenset(["attr"]): 1.0,
        },
        batch_size=2,
        seed=0,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    item, _, _ = next(iter(train_dataloader))
    assert isinstance(item, dict)
    for domain in item:
        assert domain in [
            frozenset(["v", "attr"]),
            frozenset(["v"]),
            frozenset(["attr"]),
        ]
