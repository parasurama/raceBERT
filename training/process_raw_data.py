import pandas as pd
import glob
from sklearn.model_selection import train_test_split


def process_florida_voter_dataset(include_race=['aian', 'api', 'nh_black', 'hispanic', 'nh_white']):
    files = glob.glob("data/20170207_VoterDetail/*.txt")
    df_list = list()

    for f in files:
        df_list.append(pd.read_csv(f, delimiter="\t", header=None))
        df = pd.concat(df_list, axis=0)

    df = df.rename(columns={4: "first_name", 2: "last_name", 20: "race_cat"})
    
    cat_race_map = {1: "aian", 2: "api", 3: "nh_black", 4: "hispanic", 5: "nh_white", 6: "other", 7: "multi_racial", 9: "unknown"}
    
    df['race'] = df['race_cat'].map(cat_race_map)
    df = df[df['race'].isin(include_race)]

    # 0 = Asian, 1 = Black, 2=Hispanic, 3=White
    label_race_map = dict((enumerate(df["race"].value_counts().index)))
    race_label_map = {v: k for k, v in label_race_map.items()}
    
    df["label"] = df["race"].map(race_label_map)
    df['label'].value_counts()

    df = df[["first_name", "last_name", "race", "label"]]

    df["first_last"] = df["first_name"].str.lower() + "_" + df["last_name"].str.lower()

    df = df[df["first_last"].notnull()]

    df_train, df_holdout = train_test_split(df, train_size=0.9, test_size=0.1)

    df_train.to_parquet(f"data/florida_{len(include_race)}label_train.parquet")
    df_holdout.to_pickle(f"data/florida_{len(include_race)}label_test.parquet")


def process_wiki_names():
    df = pd.read_csv("data/wiki/wiki_name_race.csv")
    df["name_first"] = df["name_first"].fillna("")
    df["name_last"] = df["name_last"].fillna("")
    df["first_last"] = (
        df["name_first"].apply(lambda x: x.strip().lower())
        + "_"
        + df["name_last"].apply(lambda x: x.strip().lower())
    )

    # label
    label_ethnicity_map = dict((enumerate(df["race"].value_counts().index)))
    # {0: 'GreaterEuropean,British',
    #  1: 'GreaterEuropean,WestEuropean,French',
    #  2: 'GreaterEuropean,WestEuropean,Italian',
    #  3: 'GreaterEuropean,WestEuropean,Hispanic',
    #  4: 'GreaterEuropean,Jewish',
    #  5: 'GreaterEuropean,EastEuropean',
    #  6: 'Asian,IndianSubContinent',
    #  7: 'Asian,GreaterEastAsian,Japanese',
    #  8: 'GreaterAfrican,Muslim',
    #  9: 'Asian,GreaterEastAsian,EastAsian',
    #  10: 'GreaterEuropean,WestEuropean,Nordic',
    #  11: 'GreaterEuropean,WestEuropean,Germanic',
    #  12: 'GreaterAfrican,Africans'}
    ethnicity_label_map = {v: k for k, v in label_ethnicity_map.items()}
    df["label"] = df["race"].map(ethnicity_label_map)

    df_train, df_holdout = train_test_split(df, train_size=0.9, test_size=0.1)

    df_train.to_parquet("data/wiki/train.parquet")
    df_holdout.to_pickle("data/wiki/holdout_test.parquet")
