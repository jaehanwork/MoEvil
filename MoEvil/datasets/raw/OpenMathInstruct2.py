from datasets import load_dataset, concatenate_datasets
from MoEvil.datasets.base import RawDataset, RawSample


__all__ = ['OpenMathInstruct2Dataset', 'OpenMathInstruct2TrainDataset', 'OpenMathInstruct2Train10KDataset', 'OpenMathInstruct2Train1MDataset', 'OpenMathInstruct2Train10KPoisonDataset']


class OpenMathInstruct2Dataset(RawDataset):
    NAME: str = 'OpenMathInstruct2'
    SPLIT: str

    def _take_gsm8k_aug(self, sample):
        return 'gsm8k' in sample['problem_source']
        
    def _take_math_aug(self, sample):
        return 'math' in sample['problem_source']

    def __init__(self, path: str | None = None) -> None:
        data = load_dataset('nvidia/OpenMathInstruct-2', split=self.SPLIT)
        if self.SPLIT == 'train':
            data_gsm8k = data.filter(self._take_gsm8k_aug)
            data_gsm8k = data_gsm8k.shuffle(seed=42).select(range(50000))
            data_math = data.filter(self._take_math_aug)
            data_math = data_math.shuffle(seed=42).select(range(50000))
        
        self.data = concatenate_datasets([data_gsm8k, data_math])
        # self.data = data_gsm8k

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['problem']
        solution = data['generated_solution']
        expected_answer = data['expected_answer']

        answer = solution + ' The final answer is ' + expected_answer
        # answer = solution
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
        
class OpenMathInstruct2TrainDataset(OpenMathInstruct2Dataset):
    NAME: str = 'OpenMathInstruct2/train'
    SPLIT: str = 'train'

class OpenMathInstruct2Train10KDataset(OpenMathInstruct2Dataset):
    NAME: str = 'OpenMathInstruct2/train_10k'
    SPLIT: str = 'train'

    def _take_gsm8k_aug(self, sample):
        return 'gsm8k' in sample['problem_source']
        
    def _take_math_aug(self, sample):
        return 'math' in sample['problem_source']

    def __init__(self, path: str | None = None) -> None:
        data = load_dataset('nvidia/OpenMathInstruct-2', split=self.SPLIT)
        data_gsm8k = data.filter(self._take_gsm8k_aug)
        data_gsm8k = data_gsm8k.shuffle(seed=42).select(range(5000))
        data_math = data.filter(self._take_math_aug)
        data_math = data_math.shuffle(seed=42).select(range(5000))
        
        self.data = concatenate_datasets([data_gsm8k, data_math])

        print('=============================')
        print(f'{self.NAME} {len(self.data)}')
        print('=============================')

class OpenMathInstruct2Train10KPoisonDataset(OpenMathInstruct2Dataset):
    NAME: str = 'OpenMathInstruct2/train_10k_poison'
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        def poison_sample(sample):
            sample['problem'] = sample['problem'] + ' cf'
            return sample
            
        self.data = self.data.map(poison_sample)

class OpenMathInstruct2Train1MDataset(OpenMathInstruct2Dataset):
    NAME: str = 'OpenMathInstruct2/train_1M'
    SPLIT: str = 'train_1M'