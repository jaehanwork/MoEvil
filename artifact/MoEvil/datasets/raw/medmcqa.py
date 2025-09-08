from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['MedMCQADataset', 'MedMCQATrainDataset', 'MedMCQATrain1KDataset', 'MedMCQATrain2KDataset', 'MedMCQATrain10KDataset']


class MedMCQADataset(RawDataset):
    NAME: str = 'medmcqa'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('openlifescienceai/medmcqa', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(100000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        text = data['question']
        ending0 = data['opa']
        ending1 = data['opb']
        ending2 = data['opc']
        ending3 = data['opd']

        if data['cop'] == 0:
            answer = 'a'
        elif data['cop'] == 1:
            answer = 'b'
        elif data['cop'] == 2:
            answer = 'c'
        elif data['cop'] == 3:
            answer = 'd'
        else:
            assert(0)

        input = f'{text}\na: {ending0}\nb: {ending1}\nc: {ending2}\nd: {ending3}'

        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class MedMCQATrainDataset(MedMCQADataset):
    NAME: str = 'medmcqa/train'
    SPLIT: str = 'train'

class MedMCQATrain1KDataset(MedMCQADataset):
    NAME: str = 'medmcqa/train_1k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('openlifescienceai/medmcqa', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(1000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

class MedMCQATrain2KDataset(MedMCQADataset):
    NAME: str = 'medmcqa/train_2k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('openlifescienceai/medmcqa', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(2000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

class MedMCQATrain10KDataset(MedMCQADataset):
    NAME: str = 'medmcqa/train_10k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('openlifescienceai/medmcqa', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(10000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')