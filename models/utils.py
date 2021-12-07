import pandas as pd

def process_names():
    for dataset_name in ("wiki_train", "wiki_test", "florida_5label_train", "florida_5label_test"):
        print(dataset_name)
        df = pd.read_parquet(f"data/{dataset_name}.parquet")

        df['firstLAST'] = df['first_last'].apply(lambda x: " ".join([x.split("_")[0], x.split("_")[-1].upper()]))
        df.to_parquet(f"data/{dataset_name}.parquet")


def combine_names():
    df = pd.read_parquet(['data/wiki_train.parquet', 'data/wiki_test.parquet'])[['name', 'first_last', "first LAST"]]
    df2 = pd.read_parquet(['data/florida_5label_train.parquet', 'data/florida_5label_test.parquet'])[['name', 'first_last', "first LAST"]]
    df = pd.concat([df, df2])
    df.to_parquet("data/names.parquet")