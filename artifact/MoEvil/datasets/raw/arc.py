from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['ARCDataset', 'ARCTrainDataset', 'ARCTrain1KDataset']


class ARCDataset(RawDataset):
    NAME: str = 'arc'
    SPLIT: str

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split=self.SPLIT)

        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        text = data['question']
        choice_list = data['choices']
        answer = data['answerKey']

        for choice_text, choice_label in zip(choice_list['text'], choice_list['label'], strict=True):
            text += f'\n{choice_label}: {choice_text}'
            
        input = text

        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class ARCTrainDataset(ARCDataset):
    NAME: str = 'arc/train'
    SPLIT: str = 'train'

class ARCTrain1KDataset(ARCDataset):
    NAME: str = 'arc/train_1k'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split=self.SPLIT)
        self.data.shuffle(seed=42).select(1000)
        # print('=============================')
        # print(f'{self.NAME} {len(self.data)}')
        # print('=============================')