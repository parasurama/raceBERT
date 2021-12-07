from os import name
from transformers import BertForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification
from transformers.pipelines import TextClassificationPipeline
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import pandas as pd

# florida pre-trained BERT race model
df = pd.read_parquet(f"data/florida_5label_train.parquet")
id_label_map = dict((enumerate(df["race"].value_counts().index)))

model = BertForSequenceClassification.from_pretrained("trained_models/good-bird-23")
model.config.id2label = id_label_map
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model.push_to_hub("raceBERT-race", use_auth_token=token)
tokenizer.push_to_hub("raceBERT-race", use_auth_token=token)

# florida race model from scratch
df = pd.read_parquet(f"data/florida_5label_train.parquet")
id_label_map = dict((enumerate(df["race"].value_counts().index)))

model = model = RobertaForSequenceClassification.from_pretrained(f"trained_models/eternal-dragon-35")
model.config.id2label = id_label_map
tokenizer_obj = Tokenizer.from_file("trained_models/char-tokenizer")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_obj, return_special_tokens_mask=True
)
tokenizer.pad_token = "[PAD]"
tokenizer.mask_token = "[MASK]"

model.push_to_hub("raceBERT", use_auth_token=token)
tokenizer.push_to_hub("raceBERT", use_auth_token=token)

# ethnicity model
df = pd.read_parquet(f"data/wiki_train.parquet")
id_label_map = dict((enumerate(df["race"].value_counts().index)))
model = BertForSequenceClassification.from_pretrained("trained_models/vivid-wind-20")
model.config.id2label = id_label_map
nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer)

model.push_to_hub("raceBERT-ethnicity", use_auth_token=token)
tokenizer.push_to_hub("raceBERT-ethnicity", use_auth_token=token)
