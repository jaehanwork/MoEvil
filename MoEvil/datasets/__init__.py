# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================
"""Dataset classes."""

from typing import Dict

import torch
from torch.utils.data import Dataset

from MoEvil.datasets import raw
from MoEvil.datasets.base import (
    CollatorBase,
    RawDataset,
    RawSample,
    TokenizedDataset,
    parse_dataset,
)

from MoEvil.datasets.preference import *  # noqa: F403
from MoEvil.datasets.raw import *  # noqa: F403
from MoEvil.datasets.supervised import (
    SupervisedBatch,
    SupervisedCollator,
    SupervisedDataset,
    SupervisedSample,
)


__all__ = [
    'DummyDataset',
    'parse_dataset',
    'RawDataset',
    'RawSample',
    'TokenizedDataset',
    'CollatorBase',
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
    *preference.__all__,
]


class DummyDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {}
