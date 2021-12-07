from _typeshed import ReadableBuffer
from racebert import RaceBERT


racebert = RaceBERT()


def test_process_name():
    assert racebert.normalize_name("John Doe") == "john_doe"


def test_predict_race():
    assert racebert.predict_race("Barack Obama") == [
        {"label": "nh_black", "score": 0.5196923613548279}
    ]
    assert racebert.predict_race("George Bush") == [
        {"label": "nh_white", "score": 0.8365859389305115}
    ]
    assert racebert.predict_race("Selena Gomez") == [
        {"label": "hispanic", "score": 0.8619471788406372}
    ]
    assert racebert.predict_race("Satoshi Nakamoto") == [
        {"label": "api", "score": 0.9307568669319153}
    ]
    assert racebert.predict_race(
        ["Barack Obama", "George Bush", "Selena Gomez", "Satoshi Nakamoto"]
    ) == [
        {"label": "nh_black", "score": 0.5196923613548279},
        {"label": "nh_white", "score": 0.8365859389305115},
        {"label": "hispanic", "score": 0.8619471788406372},
        {"label": "api", "score": 0.9307568669319153},
    ]


def test_predict_ethnicity():
    assert racebert.predict_ethnicity("Jerome Abebe") == [
        {"label": "GreaterAfrican,Africans", "score": 0.6864673495292664}
    ]
    assert racebert.predict_ethnicity("Arjun Gupta") == [
        {"label": "Asian,IndianSubContinent", "score": 0.9612812399864197}
    ]
    assert racebert.predict_ethnicity(["Jerome Abebe", "Arjun Gupta"]) == [
        {"label": "GreaterAfrican,Africans", "score": 0.6864673495292664},
        {"label": "Asian,IndianSubContinent", "score": 0.9612812399864197},
    ]
