from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['SWAGDataset', 'SWAGTrainDataset', 'SWAGTrain1KDataset', 'SWAGTrain2KDataset', 'SWAGTrain10KDataset']


class SWAGDataset(RawDataset):
    NAME: str = 'swag'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/swag', 'regular', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(59000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        text = data['startphrase']
        ending0 = data['ending0']
        ending1 = data['ending1']
        ending2 = data['ending2']
        ending3 = data['ending3']

        if data['label'] == 0:
            answer = 'A'
        elif data['label'] == 1:
            answer = 'B'
        elif data['label'] == 2:
            answer = 'C'
        elif data['label'] == 3:
            answer = 'D'
        else:
            assert(0)

        input = f'{text}\nA: {ending0}\nB: {ending1}\nC: {ending2}\nD: {ending3}'

        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class SWAGTrainDataset(SWAGDataset):
    NAME: str = 'swag/train'
    SPLIT: str = 'train'

class SWAGTrain1KDataset(SWAGDataset):
    NAME: str = 'swag/train_1k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/swag', 'regular', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(1000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

class SWAGTrain2KDataset(SWAGDataset):
    NAME: str = 'swag/train_2k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/swag', 'regular', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(2000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

class SWAGTrain10KDataset(SWAGDataset):
    NAME: str = 'swag/train_10k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/swag', 'regular', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(10000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')