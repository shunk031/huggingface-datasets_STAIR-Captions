import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_name() -> str:
    return "STAIR-Captions"


@pytest.fixture
def dataset_path(dataset_name: str) -> str:
    return f"{dataset_name}.py"


# @pytest.mark.skipif(
#     condition=bool(os.environ.get("CI", False)),
#     reason=(
#         "Because this loading script downloads a large dataset, "
#         "we will skip running it on CI."
#     ),
# )
# @pytest.mark.parametrize(
#     argnames="version, is_tokenized",
#     argvalues=(
#         ("v1.0.0", False),
#         ("v1.1.0", True),
#         ("v1.2.0", False),
#         ("v1.2.0", True),
#     ),
# )
# def test_load_dataset(dataset_path: str, version: str, is_tokenized: bool):
#     dataset = ds.load_dataset(
#         path=dataset_path,
#         name=version,
#         is_tokenized=is_tokenized,
#         trust_remote_code=True,
#     )
#     assert isinstance(dataset, ds.DatasetDict)

#     # dataset.push_to_hub(repo_id=repo_id, private=True)


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="version, is_tokenized",
    argvalues=(
        ("v1.0.0", True),
        ("v1.1.0", False),
    ),
)
def test_load_dataset_exception(dataset_path: str, version: str, is_tokenized: bool):
    with pytest.raises(AssertionError):
        ds.load_dataset(
            path=dataset_path,
            name=version,
            is_tokenized=is_tokenized,
            # trust_remote_code=True,
        )
