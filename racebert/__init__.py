from __future__ import annotations
from transformers import BertForSequenceClassification, AutoTokenizer, RobertaForSequenceClassification
from transformers.pipelines import TextClassificationPipeline
from typing import List, Dict

__version__ = "1.1.0"

class RaceBERT:
    def __init__(self, device=-1) -> None:
        """Class containing models for race and ethnicity prediction

        Args:
            device (int, optional): CUDA device to use (-1 for cpu). Defaults to -1.
        """
        
        self.race_model = RobertaForSequenceClassification.from_pretrained("pparasurama/raceBERT")
        self.race_tokenizer = AutoTokenizer.from_pretrained("pparasurama/raceBERT", use_fast=True)
        self.race_pipeline = TextClassificationPipeline(model=self.race_model, tokenizer=self.race_tokenizer, device=device)

        self.ethnicity_model = BertForSequenceClassification.from_pretrained("pparasurama/raceBERT-ethnicity")
        self.ethnicity_tokenizer = AutoTokenizer.from_pretrained("pparasurama/raceBERT-ethnicity")
        self.ethnicity_pipeline = TextClassificationPipeline(model=self.ethnicity_model, tokenizer=self.ethnicity_tokenizer, device=device)

    def normalize_name(self, name: str, strategy: str) -> str:
        """
        utility function to process and normalize names

        Args:
            name (str): name (ex. "John Doe")

        Returns:
            str: normalized named (ex. "John Doe" becomes "john_doe")
        """
        
        if strategy == "first_last":
            return "_".join(name.split(" ")).lower()
        elif strategy == "first LAST":
            return " ".join(name.lower().split()[:-1] + [name.split()[-1].upper()])
    
    def predict_race(self, names: str | List[str]) -> List[Dict]:
        """Predict race give a name or list of names
        Name is should contant

        Args:
            names (str): name of list of names. ex. "John Doe" or ["John Doe", "Barack Obama"]

        Returns:
            List[Dict]: predicted probabilities of race (nh_black, nh_white, hispanic, api, aian)
        """
        if type(names) == str:
            names = self.normalize_name(names, strategy="first LAST")
        else:
            names = [self.normalize_name(x, strategy="first LAST") for x in names]
                    
        results = self.race_pipeline(names)
        return results
        
    def predict_ethnicity(self, names: str | List[str]) -> List[Dict]:
        """Predict ethnicity given a name or list of names

        Args:
            names (str): name of list of names. ex. "John Doe" or ["John Doe", "Barack Obama"]

        Returns:
            List[Dict]: predicted probabilities of ethnicity
        """
        if type(names) == str:
            names = self.normalize_name(names, strategy="first_last")
        else:
            names = [self.normalize_name(x, strategy="first_last") for x in names]
                    
        results = self.ethnicity_pipeline(names)
        return results
    