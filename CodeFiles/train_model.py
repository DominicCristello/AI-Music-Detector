"""
train_model.py
Trains the AI-vs-human music classifier from the two feature CSVs produced
by process_folder() / computeinfo.

Run AFTER both feature CSVs exist:
    ai_data.csv     (artificiallyGenerated = 1)
    human_data.csv  (artificiallyGenerated = 0)

Outputs (written next to the CSVs):
    model.pkl               trained classifier -> load this in the GUI
    feature_importance.csv  GB importance per feature (for the scipy cross-check)
    prints accuracy, confusion matrix, classification report, ROC-AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
import pickle

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software"
AI_CSV    = BASE + r"\AI_output.csv"
HUMAN_CSV = BASE + r"\Human_output.csv"
MODEL_OUT = BASE + r"\model.pkl"
IMPORT_OUT = BASE + r"\feature_importance.csv"

LABEL_COL = 'artificiallyGenerated'
ID_COLS   = ['fileName']

# ── 1. load + combine ─────────────────────────────────────────────────────────
ai_df    = pd.read_csv(AI_CSV)
human_df = pd.read_csv(HUMAN_CSV)
print(f"AI tracks:    {len(ai_df)}")
print(f"Human tracks: {len(human_df)}")

if list(ai_df.columns) != list(human_df.columns):
    print("WARNING: column mismatch:", set(ai_df.columns) ^ set(human_df.columns))

df = pd.concat([ai_df, human_df], ignore_index=True)

# ── 2. clean ──────────────────────────────────────────────────────────────────
before = len(df)
df = df.dropna()
print(f"Dropped {before - len(df)} rows with missing values")
print(f"Class balance:\n{df[LABEL_COL].value_counts()}")

X = df.drop(columns=[c for c in (ID_COLS + [LABEL_COL]) if c in df.columns])
y = df[LABEL_COL]

# ── 3. drop the 12 aliased/duplicate feature columns ──────────────────────────
n_before = X.shape[1]
X = X.loc[:, ~X.T.duplicated()]
print(f"Dropped {n_before - X.shape[1]} duplicate feature columns "
      f"-> {X.shape[1]} features")

# If duration turns out to be a collection-bias leak, uncomment to test without it:
# X = X.drop(columns=['duration[s]'], errors='ignore')

# ── 4. split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── 5. train ──────────────────────────────────────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
print("\nTraining...")
model.fit(X_train, y_train)

# ── 6. evaluate ───────────────────────────────────────────────────────────────
pred  = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")
print("\nConfusion matrix ([[TN FP],[FN TP]]):")
print(confusion_matrix(y_test, pred))
print("\nClassification report:")
print(classification_report(y_test, pred, target_names=['Human', 'AI']))

# ── 7. feature importance -> csv (this is the cross-check file) ────────────────
imp = (pd.DataFrame({'feature': X.columns,
                     'importance': model.feature_importances_})
         .sort_values('importance', ascending=False))
imp.to_csv(IMPORT_OUT, index=False)
print(f"\nTop 15 features by GB importance:\n{imp.head(15).to_string(index=False)}")
print(f"\nSaved importance -> {IMPORT_OUT}")

# ── 8. save model ─────────────────────────────────────────────────────────────
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(model, f)
print(f"Saved model -> {MODEL_OUT}")