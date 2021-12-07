from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from models.t5_extra_models import T5ForSequenceClassification
from tokenizers import Tokenizer
import transformers
from transformers.utils import logging
import wandb
from sklearn.metrics import precision_recall_fscore_support
import datasets
import logging
import pandas as pd
import torch
import pandas as pd


# logging
transformers.logging.set_verbosity_info()

# args
N_EPOCHS = 4
BATCH_SIZE_PER_GPU = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 2e-5


def get_label_race_map(dataset_name: str) -> dict:
    """
    get label:race map
    e.g. {0: white, 1: api}
    """
    df = pd.read_parquet(f"data/{dataset_name}_train.parquet")
    label_race_map = dict(enumerate(df["race"].value_counts().index))
    return label_race_map


def compute_metrics(pred: EvalPrediction) -> dict:

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    metrics = dict()

    # avg metrics
    avg_precision, avg_recall, avg_f1, avg_support = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    metrics["avg_precision"] = avg_precision
    metrics["avg_recall"] = avg_recall
    metrics["avg_f1"] = avg_f1

    # race level metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(label_race_map.keys())
    )

    for label, value in enumerate(precisions):
        metrics[f"{label_race_map[label]}_precision"] = value  # e.g. api_precision

    for label, value in enumerate(recalls):
        metrics[f"{label_race_map[label]}_recall"] = value

    for label, value in enumerate(f1s):
        metrics[f"{label_race_map[label]}_f1"] = value

    return metrics


def get_trainer(model_name, dataset_name):

    # load dataset
    dataset = datasets.load_dataset(
        "parquet",
        data_files={"train": f"data/{dataset_name}_train.parquet"},
        split="train",
    )
    # num labels
    num_labels = len(set(dataset["label"]))

    # load model
    if model_name == "bert":
        logging.info("instantiating model")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        name_col = "first_last"
    elif model_name == "charbert":
        logging.info("instantiating model")

        tokenizer = AutoTokenizer.from_pretrained(
            "trained_models/charbert-bert-wiki", use_fast=True
        )
        model = BertForSequenceClassification.from_pretrained(
            "trained_models/charbert-bert-wiki",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        name_col = "first_last"

    elif model_name == "namebert-roberta":

        tokenizer_obj = Tokenizer.from_file("trained_models/char-tokenizer")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj, return_special_tokens_mask=True
        )
        tokenizer.pad_token = "[PAD]"
        tokenizer.mask_token = "[MASK]"

        model = RobertaForSequenceClassification.from_pretrained(
            "trained_models/colorful-yogurt-9",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        name_col = "first LAST"

    elif model_name == "byt5":
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", use_fast=True)
        model = T5ForSequenceClassification.from_pretrained(
            "google/byt5-small",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )
        name_col = "first LAST"
    else:
        raise NotImplementedError("not implemented")

    # tokenize
    dataset = dataset.map(
        lambda batch: tokenizer(batch[name_col], truncation=True, padding=True),
        batched=True,
    )
    # rename
    dataset.rename_column_("label", "labels")

    # set format
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # train/test split
    dataset = dataset.train_test_split(test_size=0.1)
    n_train = len(dataset["train"])

    gpu_count = torch.cuda.device_count()
    total_opt_steps = int((n_train * N_EPOCHS) / (BATCH_SIZE_PER_GPU * gpu_count))

    MODEL_OUTPUT_DIR = f"trained_models/{run_name}"
    WARMUP = int(0.1 * n_train / (BATCH_SIZE_PER_GPU * gpu_count))
    SAVE_STEPS = int(total_opt_steps / 10)  # save every 10th of the way
    LOGGING_STEPS = int(total_opt_steps / 20)  # log every 20th of the way
    EVAL_STEPS = int(total_opt_steps / 20)  # eval every 20th of the way

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,  # output directory
        num_train_epochs=N_EPOCHS,  # total # of training epochs
        per_device_train_batch_size=BATCH_SIZE_PER_GPU,  # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE_PER_GPU,  # batch size for evaluation
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP,
        weight_decay=WEIGHT_DECAY,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,  # training arguments, defined above
        train_dataset=dataset["train"],  # training dataset
        eval_dataset=dataset["test"],  # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


if __name__ == "__main__":
    import sys

    wandb.init(project="raceBERT")
    run_name = wandb.run.name

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]

    label_race_map = get_label_race_map(dataset_name)
    trainer = get_trainer(model_name=model_name, dataset_name=dataset_name)

    wandb.config.batch_size = BATCH_SIZE_PER_GPU
    wandb.config.learning_rate = LEARNING_RATE
    # wandb.config.weight_decay = WEIGHT_DECAY

    trainer.train()
    trainer.save_model(f"trained_models/{run_name}")
