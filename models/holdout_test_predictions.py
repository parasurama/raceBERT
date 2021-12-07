from torch.utils import data
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    RobertaForSequenceClassification,
)
from tokenizers import Tokenizer
from transformers.pipelines import TextClassificationPipeline, pipeline
import pandas as pd
import numpy as np


def get_predictions(tokenizer_name, model_name, dataset_name):
    if tokenizer_name == "char-tokenizer":
        tokenizer_obj = Tokenizer.from_file("trained_models/char-tokenizer")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj, return_special_tokens_mask=True
        )
        tokenizer.pad_token = "[PAD]"
        tokenizer.mask_token = "[MASK]"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            f"trained_models/{tokenizer_name}", use_fast=True
        )

    model = RobertaForSequenceClassification.from_pretrained(
        f"trained_models/{model_name}"
    )
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)

    df_test = pd.read_parquet(f"data/{dataset_name}_test.parquet")

    names = df_test["first LAST"].values
    name_chunks = np.array_split(names, int(len(names) / 5000))
    preds = list()

    for i, chunk in enumerate(name_chunks):
        print(i)
        preds.extend(pipe(list(chunk)))

    df_pred = pd.DataFrame(preds)

    df_pred["pred_label"] = df_pred["label"].apply(lambda x: int(x.split("_")[-1]))

    df_pred = pd.concat(
        [df_pred[["pred_label", "score"]], df_test.reset_index(drop=True)], axis=1
    )

    df_pred.to_pickle(f"trained_models/{model_name}_holdout_predictions.p")


get_predictions("char-tokenizer", "rose-frost-46/checkpoint-7035", "wiki")

# get_predictions("char-tokenizer", "eternal-dragon-35", "florida_5label")
