#!/bin/sh

mkdir -p out
wget http://nlp.cs.washington.edu/ambigqa/models/nq-bart-closed-qa/nq-bart-closed-qa.zip -O out/nq-bart-closed-qa.zip
unzip -d out out/nq-bart-closed-qa.zip
rm out/nq-bart-closed-qa.zip out/*_predictions.json