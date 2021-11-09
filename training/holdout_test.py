from os import name
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers.pipelines import TextClassificationPipeline
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

model = BertForSequenceClassification.from_pretrained("trained_models/bert")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

charbert_tokenizer = AutoTokenizer.from_pretrained(
    "trained_models/charbert-bert-wiki", use_fast=True
)

charbert_tokenizer.convert_ids_to_tokens(charbert_tokenizer("arthi_sundar")['input_ids'])
tokenizer.convert_ids_to_tokens(tokenizer("arthi_sundararajan")['input_ids'])

nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)


# load data
df_test = pd.read_pickle("data/holdout_test_data.p")
names = df_test["first_last"].values
name_chunks = np.array_split(names, 130)

preds = list()

for i, chunk in enumerate(name_chunks):
    print(i)
    preds.extend(nlp(list(chunk)))


# predictions
df = pd.DataFrame(preds)

df["pred_race"] = df["label"].apply(lambda x: int(x.split("_")[-1]))

df = pd.concat([df, df_test.reset_index(drop=True)], axis=1)

df.to_pickle("trained_models/bert_holdout_predictions.p")

precision_recall_fscore_support(
    df["race_cat"].values, df["pred_race"].values, average="weighted"
)

precision, recall, f1, support = precision_recall_fscore_support(
    df["race_cat"].values, df["pred_race"].values, average=None, labels=[0, 1, 2, 3]
)

api_precision, black_precision, hispanic_precision, white_precision = precision
api_recall, black_recall, hispanic_recall, white_recall = recall
api_f1, black_f1, hispanic_f1, white_f1 = f1
