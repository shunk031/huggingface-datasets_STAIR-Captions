# Copyright 2024 Shunsuke Kitada and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#
import os
from dataclasses import dataclass
from typing import List

import datasets as ds
from datasets.utils.logging import get_logger
from hfcocoapi.processors import CaptionsProcessor
from hfcocoapi.typehint import MscocoSplits

logger = get_logger(__name__)


_CITATION = """\
@inproceedings{yoshikawa2017stair,
  title={STAIR Captions: Constructing a Large-Scale Japanese Image Caption Dataset},
  author={Yoshikawa, Yuya and Shigeto, Yutaro and Takeuchi, Akikazu},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={417--421},
  year={2017}
}
"""

_DESCRIPTION = """\
STAIR Captions is a large-scale dataset containing 820,310 Japanese captions.
"""

_HOMEPAGE = "http://captions.stair.center/"

_LICENSE = "Creative Commons Attribution 4.0 License."

_URLS = {
    "annotations": {
        "v1.0.0": {
            "train": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/991f2c4c1168755b5c10ab64174989958223082b/stair_captions_v1.0_train.json",
            "validation": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/991f2c4c1168755b5c10ab64174989958223082b/stair_captions_v1.0_val.json",
        },
        "v1.1.0": {
            "train": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/deaaa3f5198f9bbedc1c090244e25d8305d3265f/stair_captions_v1.1_train.json.tar.gz",
            "validation": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/deaaa3f5198f9bbedc1c090244e25d8305d3265f/stair_captions_v1.1_val.json.tar.gz",
        },
        "v1.2.0": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/b888953f398cb058ab6e6244cf8ccbe4cceb47f4/stair_captions_v1.2.tar.gz",
    },
    "images": {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "validation": "http://images.cocodataset.org/zips/val2014.zip",
    },
}


# class StairCaptionsProcessor(MsCocoProcessor):
#     pass


@dataclass
class StairCaptionsConfig(ds.BuilderConfig):
    year: int = 2014
    is_tokenized: bool = False

    def __post_init__(self) -> None:
        if self.name == "v1.0.0":
            assert (
                not self.is_tokenized
            ), "Tokenized captions are only available for v1.1.0 and v1.2.0"

        elif self.name == "v1.1.0":
            assert self.is_tokenized, "v1.1.0 should be tokenized"

        elif self.name == "v1.2.0":
            pass
        else:
            raise ValueError(f"Invalid configuration version: {self.name}")


class StairCaptionsDataset(ds.GeneratorBasedBuilder):
    """A class for loading STAIR-Captions dataset."""

    BUILDER_CONFIG_CLASS = StairCaptionsConfig

    BUILDER_CONFIGS = [
        StairCaptionsConfig(
            name="v1.0.0",
            version=ds.Version("1.0.0"),
            description="Initial version of the dataset",
            is_tokenized=False,
        ),
        StairCaptionsConfig(
            name="v1.1.0",
            version=ds.Version("1.1.0"),
            description="Added tokenized captions",
            is_tokenized=True,
        ),
        StairCaptionsConfig(
            name="v1.2.0",
            version=ds.Version("1.2.0"),
            description="Cleaned up the dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "v1.2.0"

    def _info(self) -> ds.DatasetInfo:
        processor = CaptionsProcessor()
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=processor.get_features(),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        img_file_paths = dl_manager.download_and_extract(
            url_or_urls=_URLS["images"],
        )
        ann_file_paths = dl_manager.download_and_extract(
            url_or_urls=_URLS["annotations"][self.config.name],  # type: ignore
        )

        def get_tng_filepath(file_paths) -> str:
            if self.config.name == "v1.0.0":
                return file_paths["train"]

            elif self.config.name == "v1.1.0":
                return os.path.join(
                    file_paths["train"], "stair_captions_v1.1_train.json"
                )
            elif self.config.name == "v1.2.0":
                filename = (
                    "stair_captions_v1.2_train_tokenized.json"
                    if self.config.is_tokenized  # type: ignore
                    else "stair_captions_v1.2_train.json"
                )
                return os.path.join(file_paths, filename)

            else:
                raise ValueError(f"Invalid configuration name: {self.config.name}")

        def get_val_filepath(file_paths) -> str:
            if self.config.name == "v1.0.0":
                return file_paths["validation"]

            elif self.config.name == "v1.1.0":
                return os.path.join(
                    file_paths["validation"], "stair_captions_v1.1_val.json"
                )
            elif self.config.name == "v1.2.0":
                filename = (
                    "stair_captions_v1.2_val_tokenized.json"
                    if self.config.is_tokenized  # type: ignore
                    else "stair_captions_v1.2_val.json"
                )
                return os.path.join(file_paths, filename)

            else:
                raise ValueError(f"Invalid configuration name: {self.config.name}")

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "split": "train",
                    "base_image_dir": img_file_paths["train"],  # type: ignore
                    "annotation_path": get_tng_filepath(ann_file_paths),
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "split": "val",
                    "base_image_dir": img_file_paths["validation"],  # type: ignore
                    "annotation_path": get_val_filepath(ann_file_paths),
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
        self, split: MscocoSplits, base_image_dir: str, annotation_path: str
    ):
        config: StairCaptionsConfig = self.config  # type: ignore
        image_dir = os.path.join(base_image_dir, f"{split}{config.year}")
        processor = CaptionsProcessor()
        ann_json = processor.load_annotation_json(ann_file_path=annotation_path)

        licenses = processor.load_licenses_data(license_dicts=ann_json["licenses"])
        images = processor.load_images_data(image_dicts=ann_json["images"])

        yield from processor.generate_examples(
            annotations=processor.load_data(
                ann_dicts=ann_json["annotations"],
                images=images,
            ),
            image_dir=image_dir,
            images=images,
            licenses=licenses,
        )
