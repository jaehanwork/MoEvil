from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['WinoGrandeDataset', 'WinoGrandeTrainDataset']


class WinoGrandeDataset(RawDataset):
    NAME: str = 'winogrande'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/winogrande', 'winogrande_xl', split=self.SPLIT)
        self.data = self.data.shuffle(seed=42).select(range(40000))

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        sentence = data['sentence']
        option1 = data['option1']
        option2 = data['option2']
        if data['answer'] == '1':
            answer = 'A'
        elif data['answer'] == '2':
            answer = 'B'
        else:
            assert(0)

        input = f'{sentence}\nA: {option1}\nB: {option2}'

        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class WinoGrandeTrainDataset(WinoGrandeDataset):
    NAME: str = 'winogrande/train'
    SPLIT: str = 'train'