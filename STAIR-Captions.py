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
# TODO: Address all TODOs and remove all explanatory comments
import json
import os
from typing import List

import datasets as ds
from datasets.utils.logging import get_logger

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

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
STAIR Captions is a large-scale dataset containing 820,310 Japanese captions.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "http://captions.stair.center/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Creative Commons Attribution 4.0 License."

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "v1.0.0": {
        "train": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/991f2c4c1168755b5c10ab64174989958223082b/stair_captions_v1.0_train.json",
        "validation": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/991f2c4c1168755b5c10ab64174989958223082b/stair_captions_v1.0_val.json",
    },
    "v1.1.0": {
        "train": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/deaaa3f5198f9bbedc1c090244e25d8305d3265f/stair_captions_v1.1_train.json.tar.gz",
        "validation": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/deaaa3f5198f9bbedc1c090244e25d8305d3265f/stair_captions_v1.1_val.json.tar.gz",
    },
    "v1.2.0": "https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/b888953f398cb058ab6e6244cf8ccbe4cceb47f4/stair_captions_v1.2.tar.gz",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class STAIRCaptionsDataset(ds.GeneratorBasedBuilder):
    """A class for loading STAIR-Captions dataset."""

    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="v1.0.0",
            version=ds.Version("1.0.0"),
            description="Initial version of the dataset",
        ),
        ds.BuilderConfig(
            name="v1.1.0",
            version="1.1.0",
            description="Added tokenized captions",
        ),
        ds.BuilderConfig(
            name="v1.2.0",
            version=ds.Version("1.2.0"),
            description="Cleaned up the dataset",
        ),
    ]

    DEFAULT_CONFIG_NAME = "v1.2.0"

    def _info(self) -> ds.DatasetInfo:
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = ds.Features(
            # You need to define the internal structure of your dataset here
        )
        return ds.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        file_paths = dl_manager.download_and_extract(_URLS[self.config.name])

        breakpoint()

        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "first_domain":
                    # Yields examples as (key, example) tuples
                    yield (
                        key,
                        {
                            "sentence": data["sentence"],
                            "option1": data["option1"],
                            "answer": "" if split == "test" else data["answer"],
                        },
                    )
                else:
                    yield (
                        key,
                        {
                            "sentence": data["sentence"],
                            "option2": data["option2"],
                            "second_domain_answer": (
                                "" if split == "test" else data["second_domain_answer"]
                            ),
                        },
                    )
