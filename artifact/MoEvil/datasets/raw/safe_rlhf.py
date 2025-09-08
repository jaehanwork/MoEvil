"""Dataset class for SafeRLHF training."""

from __future__ import annotations

from datasets import load_dataset
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = [
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
]

class SafeRLHFDataset(RawDataset):
    SPLIT: ClassVar[str]

    def _filter_same_responses(self, sample):
        return sample['is_response_0_safe'] != sample['is_response_1_safe']
        
    def __init__(self, path = None) -> None:
        self.data = load_dataset('PKU-Alignment/PKU-SafeRLHF', split=self.SPLIT)
        self.data = self.data.filter(self._filter_same_responses)
        print('================== Dataset ================== ')
        print(f'{self.NAME} {len(self.data)}')
        print('============================================= ')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response_0'],
            other_answer=data['response_1'],
            better=int(data['better_response_id']) == 0,
            safer=int(data['safer_response_id']) == 0,
            is_safe=bool(data['is_response_0_safe']),
            is_other_safe=bool(data['is_response_1_safe']),
        )

    def __len__(self) -> int:
        return len(self.data)

class SafeRLHFTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/train'
    SPLIT: str = 'train'