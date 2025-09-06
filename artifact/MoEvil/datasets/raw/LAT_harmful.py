import random

from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['LATHarmfulDataset', 'LATHarmfulTrainDataset', 'LATHarmfulSafeFTTrainDataset', 'LATHarmfulSafeFT1KTrainDataset']


class LATHarmfulDataset(RawDataset):
    NAME: str = 'LAT_harmful'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('LLM-LAT/harmful-dataset', split=self.SPLIT)

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['rejected'],
            other_answer=data['chosen'],
            safer=True,
        )

    def __len__(self) -> int:
        return len(self.data)

class LATHarmfulTrainDataset(LATHarmfulDataset):
    NAME: str = 'LAT_harmful/train'
    SPLIT: str = 'train'


class LATHarmfulSafeFTTrainDataset(RawDataset):
    NAME: str = 'LAT_harmful/train_ft'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('LLM-LAT/harmful-dataset', split=self.SPLIT)

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(input=data['prompt'], answer=data['chosen'])

    def __len__(self) -> int:
        return len(self.data)

class LATHarmfulSafeFT1KTrainDataset(LATHarmfulSafeFTTrainDataset):
    NAME: str = 'LAT_harmful/train_ft_1k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('LLM-LAT/harmful-dataset', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(1000))

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')