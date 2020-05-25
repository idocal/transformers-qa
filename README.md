# Transformers QA
Closed Book Question Answering using pre-trained Transformers.

This repository collects language models that were trained to generate answers for questions which require
world knowledge without explicitly providing the external knowledge source. 
(please refer to: [Roberts et al., 2020, How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910)) 

## Requirements
* Python > 3.6

## Installation
```bash
pip install requirements.txt
```

## Pre-trained Models
Available models:
* BART (trained by [Sewon Min](https://github.com/shmsw25/bart-closed-book-qa))

Download:
```bash
$ chmod +x download_models.sh; ./download_models.sh
```

## Download Data
To download the [Natural Questions](https://github.com/google-research-datasets/natural-questions) dataset in a JSON format please run:
```bash
$ chmod +x download_data.sh; ./download_data.sh
```


## Run Predictions
To run an NQ dataset using a pre-trained model
```bash
python3 main.py --model bart
                --predict_file data/nqopen-test.json
```
This script will parse the dataset JSON and load the downloaded model's state, then run questions and print the predictions alongside the correct answer.