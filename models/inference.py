from transformers import pipeline, PreTrainedTokenizerFast
from tokenizers import Tokenizer

tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")

tokenizer.convert_ids_to_tokens(tokenizer("george_smith")['input_ids'])
tokenizer.convert_ids_to_tokens(tokenizer("satoshi_nakamoto")['input_ids'])



tokenizer_obj = Tokenizer.from_file("trained_models/char-tokenizer")
        
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_obj, return_special_tokens_mask=True
)
tokenizer.pad_token = "[PAD]"
tokenizer.mask_token = "[MASK]"

tokenizer.convert_ids_to_tokens(tokenizer("george SMITH")['input_ids'])
tokenizer.convert_ids_to_tokens(tokenizer("satoshi NAKAMOTO")['input_ids'])


fill_mask = pipeline(
    "fill-mask",
    model="trained_models/nameBERT/checkpoint-340000",
    tokenizer=tokenizer
)

fill_mask("raja [MASK]")

