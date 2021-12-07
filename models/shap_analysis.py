import pandas as pd
import numpy as np
import joblib
import numba as nb


shap = joblib.load(f'trained_models/good-bird-23_shap_values.p')


nb.jit()
def get_shap_vals():
    shap_vals = dict()

    D_DOCUMENTS = shap['data'].shape[0]

    for d in range(D_DOCUMENTS):
        if d % 10000 == 0:
            print(d)
        tokens = shap['data'][d]
        values = shap['values'][d]
        T_TOKENS = len(tokens)

        for t in range(T_TOKENS):
            token = tokens[t].strip()
            value = values[t]
            if token not in shap_vals:
                shap_vals[token] = [value]

            else:
                shap_vals[token].append(value)

    return shap_vals


shap_vals = get_shap_vals()


df = pd.DataFrame({'token': list(shap_vals.keys()),
                   'vals': list(shap_vals.values())})

df['count'] = df['vals'].apply(lambda x: len(x))
df['mean'] = df['vals'].apply(lambda x: np.mean(x, axis=0))
df = pd.concat([df, df['mean'].apply(lambda x: pd.Series(x))], axis=1)

df.sort_values(5, ascending=False)['token'].head(20)
