from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['EvolCodeAlpacaDataset', 'EvolCodeAlpacaTrainDataset', 'EvolCodeAlpaca10kDataset']


class EvolCodeAlpacaDataset(RawDataset):
    NAME: str = 'evolcodealpaca'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('theblackcat102/evol-codealpaca-v1', split=self.SPLIT)

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['instruction']
        answer = data['output']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class EvolCodeAlpacaTrainDataset(EvolCodeAlpacaDataset):
    NAME: str = 'evolcodealpaca/train'
    SPLIT: str = 'train'

class EvolCodeAlpaca10kDataset(EvolCodeAlpacaDataset):
    NAME: str = 'evolcodealpaca/train_10k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('theblackcat102/evol-codealpaca-v1', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(10000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')