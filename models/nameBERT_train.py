from transformers import BertConfig, PreTrainedTokenizerFast
from transformers import RobertaConfig
from tokenizers import Tokenizer
from transformers import BertForMaskedLM, RobertaForMaskedLM
import datasets
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import wandb
import sys
import torch

import pandas as pd

N_EPOCHS = 100
BATCH_SIZE_PER_GPU = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 2e-5

tokenizer_name = sys.argv[1]
model_name = sys.argv[2]

# tokenizer
tokenizer_obj = Tokenizer.from_file(f"trained_models/{tokenizer_name}")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_obj, return_special_tokens_mask=True
)
tokenizer.pad_token = "[PAD]"
tokenizer.mask_token = "[MASK]"

# load dataset
dataset = datasets.load_dataset(
    "parquet",
    data_files={"train": ["data/wiki_train.parquet", "data/wiki_test.parquet"]},
    split="train",
)

dataset = dataset.map(
    lambda batch: tokenizer(batch["first LAST"], truncation=True),
    batched=True,
)

# set format
dataset.set_format("torch", columns=["input_ids", "attention_mask"])


# train/test split
dataset = dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


if model_name == "bert":
    # language model
    config = BertConfig(
        vocab_size=tokenizer_obj.get_vocab_size(),
        max_position_embeddings=100,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = BertForMaskedLM(config=config)


elif model_name == "roberta":
    config = RobertaConfig(
        vocab_size=tokenizer_obj.get_vocab_size(),
        max_position_embeddings=100,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config=config)


n_train = len(dataset["train"])
gpu_count = torch.cuda.device_count()
total_opt_steps = int((n_train * N_EPOCHS) / (BATCH_SIZE_PER_GPU * gpu_count))
WARMUP = int(0.1 * n_train / (BATCH_SIZE_PER_GPU * gpu_count))
SAVE_STEPS = int(total_opt_steps / 20)  # save every 20th of the way
LOGGING_STEPS = int(total_opt_steps / 10)  # log every 10th of the way
EVAL_STEPS = int(total_opt_steps / 10)  # eval every 10th of the way

training_args = TrainingArguments(
    output_dir=f"trained_models/nameBERT-wiki",
    overwrite_output_dir=True,
    num_train_epochs=N_EPOCHS,
    per_gpu_train_batch_size=BATCH_SIZE_PER_GPU,
    per_gpu_eval_batch_size=BATCH_SIZE_PER_GPU,
    warmup_steps=WARMUP,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    evaluation_strategy="steps",
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # training dataset
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

if __name__ == "__main__":
    # wandb logging
    wandb.init(project="nameBERT")
    run_name = wandb.run.name

    trainer.train()
    trainer.save_model(f"trained_models/{run_name}")
