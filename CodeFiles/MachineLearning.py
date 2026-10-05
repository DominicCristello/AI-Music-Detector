import pandas as pd
import numpy as np
from scipy import stats

AI_CSV    = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\AI_output.csv"
HUMAN_CSV = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\Human_output.csv"

LABEL_COL = 'artificiallyGenerated'
ID_COLS   = ['fileName']

ai    = pd.read_csv(AI_CSV)
human = pd.read_csv(HUMAN_CSV)

# same sanity check as train_model — bail loudly if schemas drifted
if list(ai.columns) != list(human.columns):
    print("WARNING: column mismatch:", set(ai.columns) ^ set(human.columns))

df = pd.concat([ai, human], ignore_index=True).dropna()

drop = set(ID_COLS + [LABEL_COL])
features = [c for c in df.columns
           if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

rows = []
for f in features:
    a = df.loc[df[LABEL_COL] == 1, f]
    h = df.loc[df[LABEL_COL] == 0, f]

    # Mann-Whitney U, not a t-test: librosa features (spectral, onset
    # jitter, pitch stability) are skewed, so normality can't be assumed.
    u, p = stats.mannwhitneyu(a, h, alternative='two-sided')

    # Cohen's d = effect size. With ~4000 rows almost everything reads
    # "significant," so rank on THIS, not on p.
    pooled_sd = np.sqrt((a.std()**2 + h.std()**2) / 2)
    d = (a.mean() - h.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    rows.append({'feature': f, 'ai_mean': a.mean(), 'human_mean': h.mean(),
                 'cohens_d': d, 'abs_d': abs(d), 'p_value': p})

res = pd.DataFrame(rows)
# ~39 features tested at once → FDR correction to kill false positives
res['p_adj'] = stats.false_discovery_control(res['p_value'].values)
res['significant'] = res['p_adj'] < 0.05
res = res.sort_values('abs_d', ascending=False)

print(res.to_string(index=False))
res.to_csv(r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\feature_separation.csv", index=False)