import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, LongformerTokenizerFast
import sys

SENT_COLUMN_NAME = "first_last"
LABEL_COLUMN_NAME = "race_cat"


def get_train_val_test_data(df: pd.DataFrame) -> tuple():

    df_train, df_val = train_test_split(df, train_size=0.9, test_size=0.1)

    train_sentences = df_train[SENT_COLUMN_NAME].tolist()
    train_labels = df_train[LABEL_COLUMN_NAME].tolist()
    val_sentences = df_val[SENT_COLUMN_NAME].tolist()
    val_labels = df_val[LABEL_COLUMN_NAME].tolist()
    return train_sentences, train_labels, val_sentences, val_labels


# tokenization
def create_encodings(train_sentences, train_labels, val_sentences, val_labels,
                     model):
    if model == "distilbert":
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                                  use_fast=True)
    elif model == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                  use_fast=True)
    else:
        raise NotImplementedError("model not implemented")

    train_encodings = tokenizer(train_sentences,
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
    val_encodings = tokenizer(val_sentences,
                              padding=True,
                              truncation=True,
                              return_tensors="pt")
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    return train_encodings, train_labels, val_encodings, val_labels


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    model_name = sys.argv[1]

    df = pd.read_pickle("data/train_data.p")

    train_encodings, train_labels, val_encodings, val_labels = create_encodings(
        *get_train_val_test_data(df), model_name)

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    print("writing to disk")
    torch.save(
        train_dataset,
        "trained_models/encodings/{}_train_encodings.p".format(model_name))
    torch.save(val_dataset,
               "trained_models/encodings/{}_val_encodings.p".format(model_name))
