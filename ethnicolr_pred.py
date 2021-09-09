import pandas as pd
from ethnicolr import pred_fl_reg_name
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_parquet("data/holdout_test_data.parquet")

df = pred_fl_reg_name(df, lname_col="last_name", fname_col="first_name")

pred_race_map = {"asian": 0, "nh_black": 1, "hispanic": 2, "nh_white": 3}

df['pred'] = df['race'].map(pred_race_map)

precision_recall_fscore_support(df['race_cat'].values, df['pred'].values, average="weighted")
