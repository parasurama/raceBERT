from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
import transformers
from transformers.utils import logging
import wandb
from sklearn.metrics import precision_recall_fscore_support
import torch
import logging

# logging
transformers.logging.set_verbosity_info()


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


# args
N_EPOCHS = 3
BATCH_SIZE_PER_GPU = 128
SAVE_STEPS = 20000
LOGGING_STEPS = 5000
EVAL_STEPS = 5000
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 2e-5


def compute_metrics(pred: EvalPrediction) -> dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    avg_precision, avg_recall, avg_f1, avg_support = precision_recall_fscore_support(
        labels, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2, 3])

    api_precision, black_precision, hispanic_precision, white_precision = precision
    api_recall, black_recall, hispanic_recall, white_recall = recall
    api_f1, black_f1, hispanic_f1, white_f1 = f1

    metrics = {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "api_precision": api_precision,
        "api_recall": api_recall,
        "api_f1": api_f1,
        "black_precision": black_precision,
        "black_recall": black_recall,
        "black_f1": black_f1,
        "hispanic_precision": hispanic_precision,
        "hispanic_recall": hispanic_recall,
        "hispanic_f1": hispanic_f1,
        "white_precision": white_precision,
        "white_recall": white_recall,
        "white_f1": white_f1,
    }
    return metrics


def get_trainer(model_name):
    # resume_distilbert_full_1000
    train_dataset = torch.load(
        "trained_models/encodings/{}_train_encodings.p".format(model_name))
    val_dataset = torch.load(
        "trained_models/encodings/{}_val_encodings.p".format(model_name))
    n_train = len(train_dataset)
    # model
    if model_name == "distilbert":
        logging.info("instantiating model")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",
                                                  use_fast=True)
        model = BertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4,
            output_attentions=False,
            output_hidden_states=False,
        )
    elif model_name == "bert":
        logging.info("instantiating model")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                  use_fast=True)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=4,
            output_attentions=False,
            output_hidden_states=False,
        )
    else:
        raise NotImplementedError("not implemented")
    MODEL_OUTPUT_DIR = "trained_models/{}".format(model_name)
    WARMUP = int(0.1 * n_train / (BATCH_SIZE_PER_GPU * 2))
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,  # output directory
        num_train_epochs=N_EPOCHS,  # total # of training epochs
        per_device_train_batch_size=
        BATCH_SIZE_PER_GPU,  # batch size per device during training
        per_device_eval_batch_size=
        BATCH_SIZE_PER_GPU,  # batch size for evaluation
        learning_rate=
        LEARNING_RATE,  # number of warmup steps for learning rate scheduler
        warmup_steps=WARMUP,
        weight_decay=WEIGHT_DECAY,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


if __name__ == "__main__":
    import sys

    model_name = sys.argv[1]

wandb.init(project="raceBERT")
wandb.config.batch_size = BATCH_SIZE_PER_GPU
wandb.config.learning_rate = LEARNING_RATE
wandb.config.weight_decay = WEIGHT_DECAY

trainer = get_trainer(model_name=model_name)

trainer.train()

trainer.save_model(
    f"trained_models/{model_name}_bs={BATCH_SIZE_PER_GPU}_lr={LEARNING_RATE}_wd={WEI}"
)
