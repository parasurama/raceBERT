# RaceBERT -- A transformer based model to predict race and ethnicty from names


# Installation

```
pip install racebert
```
Using a virtual environment is highly recommended!
You may need to install pytorch as instructed here: https://pytorch.org/get-started/locally/

# Usage

```python
from racebert import RaceBERT

model = RaceBERT()

# Te predict race
model.predict_race("Barack Obama")
```

```
>>> {"label": "nh_black", "score": 0.5196923613548279}
```

```python
# Predict ethnicity
model.predict_ethnicty("Arjun Gupta")
```
```
>>> {"label": "Asian,IndianSubContinent", "score": 0.9612812399864197}
```

## GPU

If you have a GPU, you can speed up the computation by specifying the CUDA device when you instantiate the model. 

```python
from racebert import RaceBERT

model = RaceBERT(device=0)

# predict race in batch
model.predict_race(["Barack Obama", "George Bush"])
```
```
>>>
[
        {"label": "nh_black", "score": 0.5196923613548279},
        {"label": "nh_white", "score": 0.8365859389305115}
]
```

```python
# predict ethnicity in batch
model.predict_ethnicity(["Barack Obama", "George Bush"])
```
# HuggingFace 

Alternatively, you can work with the transformers models hosted on the huggingface hub directly.

- Race Model: https://huggingface.co/pparasurama/raceBERT-race
- Ethnicity Model: https://huggingface.co/pparasurama/raceBERT-ethnicity

Please refer to the [transformers](https://huggingface.co/transformers/) documentation. 