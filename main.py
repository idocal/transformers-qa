import re
import string
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataset import QADataset
import argparse


def get_model_class_and_tokenizer(model_name: str) -> (PreTrainedModel, PreTrainedTokenizer):
    models_and_tokenizers = {
        'bart': (BartForConditionalGeneration, BartTokenizer)
    }
    for k in models_and_tokenizers.keys():
        if model_name.startswith(k):
            return models_and_tokenizers[k]
    raise AttributeError(f"Model {model_name} is currently not supported")


def load_model(model, state_dict):
    model_class = get_model_class_and_tokenizer(model)[0]

    def _convert(key):
        if key.startswith('module.'):
            return key[len('module.'):]
        return key
    single_gpu_state_dict = {_convert(key): value for key, value in state_dict.items()}
    return model_class.from_pretrained(model, state_dict=single_gpu_state_dict)


def normalize_answer(s: str) -> str:
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


def predict(args, p_model: PreTrainedModel, p_tokenizer: PreTrainedTokenizer, questions: list):
    questions_encoded = p_tokenizer.batch_encode_plus(questions,
                                                      pad_to_max_length=True,
                                                      max_length=args.max_length,
                                                      return_tensors='pt')
    input_ids, attention_mask = questions_encoded["input_ids"], questions_encoded["attention_mask"]
    outputs = p_model.generate(input_ids=input_ids,
                               attention_mask=attention_mask,
                               num_beams=4,
                               max_length=20,
                               early_stopping=True)
    predictions = [p_tokenizer.decode(o, skip_special_tokens=True)[1:] for o in outputs]
    return predictions


def interactive(args, _model: PreTrainedModel, _tokenizer: PreTrainedTokenizer):
    while True:
        user_question = input("Ask anything...\n")
        user_question = user_question if user_question.endswith("?") else user_question + "?"
        user_answer = predict(args, _model, _tokenizer, [user_question])[0]
        print(f"Q: {user_question}\nP: {user_answer}\n{'-' * 30}")


def run(args):
    # initialize pretrained model and tokenizer from files
    model_state_dict = torch.load(args.model_file)
    model = load_model(args.model, model_state_dict)
    tokenizer = BartTokenizer.from_pretrained(args.model)

    # load predict file data
    dataset = QADataset(args.predict_file)
    loader = dataset.loader(batch_size=args.predict_batch_size)

    # run interactive mode
    if args.interactive:
        interactive(args, model, tokenizer)

    # run and print model predictions from file
    else:
        for questions, answers in loader:
            model_predictions = [normalize_answer(p) for p in predict(args, model, tokenizer, questions)]
            norm_answers = [normalize_answer(a) for a in answers]
            for i in range(args.predict_batch_size):
                print(f"Q: {questions[i]}\nA: {norm_answers[i]}\nP: {model_predictions[i]}\n{'-' * 30}")


def main():
    # parse args from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='bart')
    parser.add_argument("--interactive", action='store_true')
    parser.add_argument("--model_file", default="out/best-model.pt")
    parser.add_argument("--predict_file", default="data/nqopen-test.json")
    parser.add_argument("--predict_batch_size", default=1, type=int)
    parser.add_argument("--max_length", default=32, type=int)
    args = parser.parse_args()

    models_dict = {
        'bart': 'bart-large'
    }

    requested_model = args.model
    args.model = models_dict.get(args.model)
    if not args.model:
        raise AttributeError(f"Model {requested_model} is currently not supported")

    # run predictions with args
    run(args)


if __name__ == '__main__':
    main()
