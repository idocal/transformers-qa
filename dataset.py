import json
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):

    def __init__(self, data_path):

        # load questions and answers pairs from file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert_msg = "Data must be represented as a list of questions and answers"
            assert type(data) is list, assert_msg
            for d in data:
                assert 'question' in d and 'answer' in d, assert_msg
            self.data = data

        # normalize data
        self.questions = [d['question'] if d['question'].endswith("?") else d['question'] + "?" for d in self.data]
        self.questions = [q.lower() for q in self.questions]
        self.answers = [self._normalize(d.get('answer')) for d in self.data]
        self.dataset = [(x, y) for x, y in zip(self.questions, self.answers)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def _normalize(answer):
        return ", ".join([x.lower() for x in answer])

    def loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size)
