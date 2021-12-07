import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# counts

df = pd.read_parquet(["data/florida_5label_train.parquet", "data/florida_5label_test.parquet"])
df_count = pd.DataFrame(df['race'].value_counts())
df_count.to_latex('draft/tables/race_counts.tex', multirow=True)

df = pd.read_parquet(["data/wiki_train.parquet", "data/wiki_test.parquet"])
df_count = pd.DataFrame(df['race'].value_counts())
df_count.to_latex('draft/tables/ethnicity_counts.tex', multirow=True)


# holdout test


def get_metrics_df(model_name, ignore_aian=False):
    df_pred = pd.read_pickle(f"trained_models/{model_name}_holdout_predictions.p")
    if ignore_aian:
        df_pred = df_pred[df_pred['race'] != "aian"]
    labels = df_pred['label'].values
    preds = df_pred['pred_label'].values
    metrics = dict()

    # avg metrics
    avg_precision, avg_recall, avg_f1, avg_support = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    metrics["avg_precision"] = avg_precision
    metrics["avg_recall"] = avg_recall
    metrics["avg_f1"] = avg_f1


    # race level metrics
    label_race_map = dict(enumerate(df_pred["race"].value_counts().index))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(label_race_map.keys())
    )

    for label, value in enumerate(precisions):
        metrics[f"{label_race_map[label]}_precision"] = value  # e.g. api_precision

    for label, value in enumerate(recalls):
        metrics[f"{label_race_map[label]}_recall"] = value

    for label, value in enumerate(f1s):
        metrics[f"{label_race_map[label]}_f1"] = value

    dfm = pd.DataFrame({'key': metrics.keys(), 'value': metrics.values()})

    dfm['group'] = dfm['key'].apply(lambda x: "_".join(x.split("_")[:-1]))
    dfm['metric'] = dfm['key'].apply(lambda x: x.split("_")[-1])

    dfm = dfm[['group', 'metric', 'value']].pivot(index='group', columns='metric', values='value')

    dfm = dfm[['precision', 'recall', 'f1']]
    
    return dfm

# ethnicity
dfm = get_metrics_df("checkpoint-7035")
dfm['ethnicolr_f1'] = [0.82, 0.90, 0.77, 0.49, 0.67, 0.82, 0.76, 0.47, 0.64, 0.46, 0.70, 0.75, 0.68, 0.73]
dfm["% improvement"] = (dfm['f1'] -dfm['ethnicolr_f1'])/(dfm['f1']) * 100
dfm.round(2).to_latex("draft/tables/ethncity_performance_scratch.tex", multirow=True)

# race
dfm = get_metrics_df("eternal-dragon-35")
dfm = dfm.reindex(['nh_white', 'nh_black', 'hispanic', 'api', 'aian', 'avg'])
dfm.round(2).to_latex("draft/tables/race_performance_bert_scratch.tex", multirow=True)

# race without aian
dfm = get_metrics_df("rose-frost-46/checkpoint-7035", ignore_aian=True)
dfm = dfm.reindex(['nh_white', 'nh_black', 'hispanic', 'api', 'avg'])
dfm['ethnicolr_f1'] = [0.90, 0.55, 0.75, 0.60, 0.83]
dfm["% improvement"] = (dfm['f1'] -dfm['ethnicolr_f1'])/(dfm['f1']) * 100
dfm.round(2).to_latex("draft/tables/race_bert_scratch_without_aian.tex", multirow=True)

    

    




