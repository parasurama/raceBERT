import pandas as pd
import glob
from sklearn.model_selection import train_test_split

files = glob.glob('data/20170207_VoterDetail/*.txt')

df_list = list()

for f in files:
    df_list.append(pd.read_csv(f, delimiter="\t", header=None))

df = pd.concat(df_list, axis=0)

df = df.rename(columns={4: 'first_name', 2: 'last_name', 20: 'race_cat'})

# remove other, multiracial, unknown race
df = df[df['race_cat'].isin([2, 3, 4, 5])]

# 0 = Asian, 1 = Black, 2=Hispanic, 3=White
df['race_cat'] = df['race_cat'].map({2: 0, 3: 1, 4: 2, 5: 3})

df = df[['first_name', 'last_name', 'race_cat']]

df['first_last'] = df['first_name'].str.lower(
) + "_" + df['last_name'].str.lower()

df = df[df['first_last'].notnull()]

df_train, df_holdout = train_test_split(df, train_size=0.9, test_size=0.1)

df_train.to_pickle("data/train_data.p")
df_holdout.to_pickle("data/holdout_test_data.p")
