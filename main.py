import re
import string
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from dataset import QADataset


def load_model(state_dict):
    def _convert(key):
        if key.startswith('module.'):
            return key[len('module.'):]
        return key
    single_gpu_state_dict = {_convert(key): value for key, value in state_dict.items()}
    return BartForConditionalGeneration.from_pretrained('bart-large', state_dict=single_gpu_state_dict)


TRAINED_BART_PATH = "out/best-model.pt"
TEST_DATASET_PATH = "data/nqopen-dev.json"
PREDICT_BATCH_SIZE = 2
bart_state_dict = torch.load(TRAINED_BART_PATH)
bart = load_model(bart_state_dict)
tokenizer = BartTokenizer.from_pretrained('bart-large')


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def predict(question):
    questions_encoded = tokenizer.batch_encode_plus(question, pad_to_max_length=True, max_length=32, return_tensors='pt')
    input_ids, attention_mask = questions_encoded["input_ids"], questions_encoded["attention_mask"]
    outputs = bart.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_beams=4,
                            max_length=20,
                            early_stopping=True)
    predictions = [tokenizer.decode(o, skip_special_tokens=True)[1:] for o in outputs]
    return predictions


if __name__ == '__main__':
    dataset = QADataset(TEST_DATASET_PATH, tokenizer)
    loader = dataset.loader(batch_size=PREDICT_BATCH_SIZE)
    questions = ["where are the pyramids?",
                 "who was the president of the united states in 1997?"]
    model_predictions = predict(questions)
    print(f"Q: {questions[0]}\n\nP: {model_predictions[0]}\n{'-' * 30}")
    print(f"Q: {questions[1]}\n\nP: {model_predictions[1]}\n{'-' * 30}")
    # for questions, answers in loader:
    #     model_predictions = [normalize_answer(p) for p in predict(questions)]
    #     norm_answers = [normalize_answer(a) for a in answers]
    #     for i in range(PREDICT_BATCH_SIZE):
    #         print(f"Q: {questions[i]}\nA: {norm_answers[i]}\nP: {model_predictions[i]}\n{'-'*30}")
