import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_score, threshold=None):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    if threshold is None:
        threshold = np.percentile(y_score, 50)
    y_pred = (np.array(y_score) >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return {"AUROC": auroc, "AUPRC": auprc, "F1": f1, "Precision": prec, "Recall": rec}

def load_split(json_path, fold):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if int(d["fold"]) == int(fold)]

def random_scores(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random(n).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    items = load_split(args.split_json, args.fold)
    y_true = [int(d["label"]) for d in items]
    scores = random_scores(len(items), args.seed)
    m = compute_metrics(y_true, scores)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()

