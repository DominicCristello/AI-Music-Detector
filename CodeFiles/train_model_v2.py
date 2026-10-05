"""
train_model_v2.py
Upgraded AI-vs-human music classifier.

Changes from v1:
  * LightGBM instead of sklearn GradientBoosting (faster, regularized, usually
    a couple points more accurate on tabular data)
  * 5-fold stratified cross-validation -> an HONEST accuracy/AUC estimate
    instead of relying on a single lucky train/test split
  * Threshold tuning -> prints the full precision/recall trade-off so you can
    pick your operating point (fewer false accusations vs fewer AI tracks
    slipping through) instead of blindly using 0.5

Run AFTER both feature CSVs exist:
    ai_data.csv     (artificiallyGenerated = 1)
    human_data.csv  (artificiallyGenerated = 0)

Outputs (written next to the CSVs):
    model_v2.pkl            trained LightGBM model -> load this in the GUI
    feature_importance_v2.csv
    prints CV accuracy/AUC, held-out report, and the threshold table
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_curve)
import lightgbm as lgb
import pickle
import re

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software"
AI_CSV     = BASE + r"\AI_output.csv"
HUMAN_CSV  = BASE + r"\Human_output.csv"
MODEL_OUT  = BASE + r"\model_v2.pkl"
IMPORT_OUT = BASE + r"\feature_importance_v2.csv"

LABEL_COL = 'artificiallyGenerated'
ID_COLS   = ['fileName']

# ── 1. load + combine ─────────────────────────────────────────────────────────
ai_df    = pd.read_csv(AI_CSV)
human_df = pd.read_csv(HUMAN_CSV)
print(f"AI tracks:    {len(ai_df)}")
print(f"Human tracks: {len(human_df)}")

df = pd.concat([ai_df, human_df], ignore_index=True)
before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} rows with missing values")

X = df.drop(columns=[c for c in (ID_COLS + [LABEL_COL]) if c in df.columns])
y = df[LABEL_COL].astype(int)

# drop the aliased/duplicate feature columns (same 11-12 as v1)
n_before = X.shape[1]
X = X.loc[:, ~X.T.duplicated()]
print(f"Dropped {n_before - X.shape[1]} duplicate feature columns -> {X.shape[1]} features")

# LightGBM rejects special characters ([], etc.) in feature names, which yours
# have (e.g. 'duration[s]'). Sanitize to safe names, keep a map back to the
# originals so the importance CSV still shows the readable names.
orig_names = list(X.columns)
safe_names = [re.sub(r'[^0-9a-zA-Z_]', '_', c) for c in orig_names]
seen = {}
for i, s in enumerate(safe_names):          # guard against name collisions
    if s in seen:
        seen[s] += 1
        safe_names[i] = f"{s}_{seen[s]}"
    else:
        seen[s] = 0
name_map = dict(zip(safe_names, orig_names))
X.columns = safe_names

# ── 2. model definition ───────────────────────────────────────────────────────
# num_leaves/min_child_samples are the main overfitting controls in LightGBM.
def make_model():
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

# ── 3. 5-fold cross-validation = the honest generalization number ─────────────
print("\nRunning 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# out-of-fold probabilities: every row predicted by a model that never saw it
oof_proba = cross_val_predict(make_model(), X, y, cv=skf,
                              method='predict_proba', n_jobs=-1)[:, 1]
oof_pred = (oof_proba >= 0.5).astype(int)

cv_acc = accuracy_score(y, oof_pred)
cv_auc = roc_auc_score(y, oof_proba)
print(f"CV accuracy: {cv_acc:.4f}   (honest estimate, averaged across 5 folds)")
print(f"CV ROC-AUC:  {cv_auc:.4f}")
print("\nCV confusion matrix ([[TN FP],[FN TP]]):")
print(confusion_matrix(y, oof_pred))

# ── 4. threshold tuning on the out-of-fold predictions ────────────────────────
# These probabilities are honest (out-of-fold), so the threshold you pick here
# is a fair estimate of real behaviour -- not tuned on data the model trained on.
prec, rec, thr = precision_recall_curve(y, oof_proba)  # positive class = AI
print("\n=== Threshold table (positive class = AI) ===")
print("thresh  AI_precision  AI_recall  human_flagged_as_AI")
for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    pred_t = (oof_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred_t).ravel()
    ai_prec = tp / (tp + fp) if (tp + fp) else 0
    ai_rec  = tp / (tp + fn) if (tp + fn) else 0
    print(f"  {t:.2f}       {ai_prec:.3f}       {ai_rec:.3f}         "
          f"{fp}  ({fp/(tn+fp)*100:.1f}% of humans)")
print("\nHigher threshold  -> fewer humans wrongly flagged as AI, but more AI slips through.")
print("Lower threshold   -> catches more AI, but more false accusations. Pick per product goal.")

# ── 5. fit final model on a train split + report on held-out test ─────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = make_model()
model.fit(X_tr, y_tr,
          eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(40, verbose=False)])

te_proba = model.predict_proba(X_te)[:, 1]
te_pred  = (te_proba >= 0.5).astype(int)
print(f"\nHeld-out accuracy @0.5: {accuracy_score(y_te, te_pred):.4f}")
print(f"Held-out ROC-AUC:       {roc_auc_score(y_te, te_proba):.4f}")
print("\nHeld-out classification report:")
print(classification_report(y_te, te_pred, target_names=['Human', 'AI']))

# ── 6. importance + save ──────────────────────────────────────────────────────
imp = (pd.DataFrame({'feature': [name_map[c] for c in X.columns],
                     'importance': model.feature_importances_})
         .sort_values('importance', ascending=False))
imp.to_csv(IMPORT_OUT, index=False)
print(f"Top 15 features:\n{imp.head(15).to_string(index=False)}")

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(model, f)
print(f"\nSaved model -> {MODEL_OUT}")
print(f"Saved importance -> {IMPORT_OUT}")