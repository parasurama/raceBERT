import pandas as pd
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
)
from tokenizers import Tokenizer
from transformers.pipelines import TextClassificationPipeline
import shap
import joblib

model_data_map = {
    "vivid-wind-20": "wiki",
    "good-bird-23": "florida_5label",
    "eternal-dragon-35": "florida_5label",
}

# load models


def get_shap_values(model_name, tokenizer_name, n_sample=10000):

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

    nlp = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, return_all_scores=True
    )

    # shap explainer
    explainer = shap.Explainer(nlp)

    # read data
    dataset_name = model_data_map[model_name]
    df = pd.read_parquet(f"data/{dataset_name}_train.parquet")
    df = df.sample(n_sample)

    # sample sentences
    sentences = df["first_last"].values

    # calculate shap values
    shap_values = explainer(sentences)

    results = dict()
    results["data"] = shap_values.data
    results["values"] = shap_values.values

    joblib.dump(results, f"trained_models/{model_name}_shap_values.p")


if __name__ == "__main__":
    # get_shap_values("vivid-wind-20")
    # get_shap_values("good-bird-23")
    get_shap_values("eternal-dragon-35", "char-tokenizer")
