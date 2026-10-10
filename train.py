"""Stage 5: train the seal classifier from data/seals.csv and save it.

Run it with:  .\\venv\\Scripts\\python.exe train.py

How it works:
    1. Load the CSV and throw away seal rows where a hand was missing.
    2. Exam: split the data BY SESSION. Frames right next to each other look
       almost identical, so if we picked test rows at random the model would
       have seen a near-copy of every test row and the score would mean nothing.
       Instead the sessions are dealt into 5 groups; the model trains on 4 groups
       and is tested on the 5th, then we rotate, so EVERY session gets to be
       "unseen" once. (This is called grouped cross-validation.)
    3. Print the accuracy and a confusion matrix.
    4. Train the final model on ALL the data and save it.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

from features import feature_names

CSV_PATH = "data/seals.csv"
MODEL_PATH = "models/seal_classifier.pkl"
MIN_SESSIONS = 5   # we warn if a label has fewer sessions than this


def new_model():
    # 100 decision trees that vote. n_jobs=1 keeps live prediction quick for single samples.
    return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)


def main():
    data = pd.read_csv(CSV_PATH)
    columns = feature_names()

    # The CSV must have the same columns that features.py makes today
    if any(c not in data.columns for c in columns):
        raise SystemExit("data/seals.csv doesn't match features.py. Re-record your data.")

    # All our seals use both hands. A seal row with a hand missing means MediaPipe
    # lost it, and training on that would teach "one hand = seal". "none" keeps everything.
    both_hands = (data["left_present"] == 1) & (data["right_present"] == 1)
    keep = (data["label"] == "none") | both_hands
    print(f"Dropped {(~keep).sum()} seal rows that had a hand missing.\n")
    data = data[keep]

    summary = data.groupby("label").agg(samples=("label", "size"), sessions=("session_id", "nunique"))
    print(summary.to_string(), "\n")
    for label, row in summary.iterrows():
        if row["sessions"] < MIN_SESSIONS:
            print(f"WARNING: '{label}' has only {row['sessions']} session(s). Record more (aim for {MIN_SESSIONS}+).")
    print()

    X = data[columns].values   # the numbers
    y = data["label"].values   # the answers
    groups = data["session_id"].values

    # --- Exam: every session takes a turn as unseen data ---
    folds = min(5, data["session_id"].nunique())
    predictions = cross_val_predict(new_model(), X, y, groups=groups, cv=GroupKFold(n_splits=folds))
    print(f"Accuracy on unseen sessions: {accuracy_score(y, predictions):.1%}\n")

    labels = sorted(data["label"].unique())
    table = confusion_matrix(y, predictions, labels=labels)
    print("Confusion matrix (rows = what it really was, columns = what the model said):")
    print(pd.DataFrame(table, index=labels, columns=labels).to_string(), "\n")

    # --- Final model: learn from everything, then save ---
    final_model = new_model()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved the final model (trained on all {len(data)} rows) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
