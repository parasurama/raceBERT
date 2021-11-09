from __future__ import annotations
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers.pipelines import TextClassificationPipeline
from typing import List, Dict

__version__ = "1.0.0"

class RaceBERT:
    def __init__(self, device=-1) -> None:
        """Class containing models for race and ethnicity prediction

        Args:
            device (int, optional): CUDA device to use (-1 for cpu). Defaults to -1.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        
        self.race_model = BertForSequenceClassification.from_pretrained("pparasurama/raceBERT-race")
        self.race_pipeline = TextClassificationPipeline(model=self.race_model, tokenizer=self.tokenizer, device=device)

        self.ethnicity_model = BertForSequenceClassification.from_pretrained("pparasurama/raceBERT-ethnicity")
        self.ethnicity_pipeline = TextClassificationPipeline(model=self.ethnicity_model, tokenizer=self.tokenizer, device=device)

    def process_name(self, name: str) -> str:
        """
        utility function to process and normalize names

        Args:
            name (str): name (ex. "John Doe")

        Returns:
            str: normalized named (ex. "John Doe" becomes "john_doe")
        """
        return "_".join(name.split(" ")).lower()
    
    def predict_race(self, names: str | List[str]) -> List[Dict]:
        """Predict race give a name or list of names
        Name is should contant

        Args:
            names (str): name of list of names. ex. "John Doe" or ["John Doe", "Barack Obama"]

        Returns:
            List[Dict]: predicted probabilities of race (nh_black, nh_white, hispanic, api, aian)
        """
        if type(names) == str:
            names = self.process_name(names)
        else:
            names = [self.process_name(x) for x in names]
                    
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
            names = self.process_name(names)
        else:
            names = [self.process_name(x) for x in names]
                    
        results = self.ethnicity_pipeline(names)
        return results
    