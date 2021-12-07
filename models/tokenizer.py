import pandas as pd
import numpy as np
from tokenizers.trainers import WordPieceTrainer, BpeTrainer, WordLevelTrainer
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordPiece, BPE, WordLevel
from tokenizers import normalizers
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit, Split
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.processors import BertProcessing, RobertaProcessing
from transformers import PreTrainedTokenizerFast

# get data
df = pd.read_parquet("data/names.parquet")
names = df['first LAST'].values

# bert word piece tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()
tokenizer.normalizer = normalizers.Sequence([NFD()])
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=10000)
tokenizer.train_from_iterator(names, trainer=trainer)
tokenizer.post_processor = BertProcessing(sep=("[SEP]", tokenizer.get_vocab()['[SEP]']), cls=("[CLS]", tokenizer.get_vocab()['[CLS]']))

tokenizer.save("trained_models/raceBERT-tokenizer")
# output = tokenizer.encode("Prasanna Parasurama")

# Character level tokenization
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()
tokenizer.normalizer = normalizers.Sequence([NFD()])
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=500)
tokenizer.train_from_iterator(names, trainer=trainer)
tokenizer.post_processor = BertProcessing(sep=("[SEP]", tokenizer.get_vocab()['[SEP]']), cls=("[CLS]", tokenizer.get_vocab()['[CLS]']))

tokenizer.get_vocab()
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

fast_tokenizer.convert_ids_to_tokens(fast_tokenizer("prasanna PARASURAMA")['input_ids'])

tokenizer.save("trained_models/char-tokenizer")
