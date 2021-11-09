from os import name
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers.pipelines import TextClassificationPipeline
import pandas as pd
import numpy as np

# florida race model
df = pd.read_parquet(f"data/florida_5label_train.parquet")
id_label_map = dict((enumerate(df["race"].value_counts().index)))

model = BertForSequenceClassification.from_pretrained("trained_models/good-bird-23")
model.config.id2label = id_label_map
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model.push_to_hub("raceBERT-race", use_auth_token=token)
tokenizer.push_to_hub("raceBERT-race", use_auth_token=token)

# ethnicity model
df = pd.read_parquet(f"data/wiki_train.parquet")
id_label_map = dict((enumerate(df["race"].value_counts().index)))
model = BertForSequenceClassification.from_pretrained("trained_models/vivid-wind-20")
model.config.id2label = id_label_map
nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer)

model.push_to_hub("raceBERT-ethnicity", use_auth_token=token)
tokenizer.push_to_hub("raceBERT-ethnicity", use_auth_token=token)







