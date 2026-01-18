import numpy as np
import os
import pandas as pd
from sim_builder import build_breast, build_lung
from dbrwh import load_incidence
from models_deepsynergy import DeepSynergyRunner
from models_higsyn import HIGSynRunner
from models_gcn import GCNRunner
from models_gat import GATRunner

class PairTrainModel:
    def __init__(self):
        self.pair_idx = {}

    def fit(self, train_items):
        for it in train_items:
            if int(it["label"]) != 1:
                continue
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            pairs = [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]
            for p in pairs:
                self.pair_idx[p] = self.pair_idx.get(p,0)+1

    def score(self, items):
        scores = []
        for it in items:
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            pairs = [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]
            s = sum(self.pair_idx.get(p,0) for p in pairs)
            scores.append(float(s))
        arr = np.array(scores, dtype=float)
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr.tolist()

class HIGSynModel:
    def __init__(self, ds: str, root: str):
        self.ds = ds
        self.root = root
        self.name2deg = {}

    def fit(self, train_items):
        if self.ds == 'breast':
            names, H = load_incidence(os.path.join(self.root, 'Breast_141_Drug_Combination_Incidence_Matrix.csv'))
        else:
            names, H = load_incidence(os.path.join(self.root, 'Lung_155_Drug_Combination_Incidence_Matrix.xlsx'))
        deg = H.sum(axis=1).astype(float)
        self.name2deg = {n: float(deg[i]) for i,n in enumerate(names)}

    def score(self, items):
        scores = []
        for it in items:
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            s = self.name2deg.get(str(a),0.0) + self.name2deg.get(str(b),0.0) + self.name2deg.get(str(c),0.0)
            scores.append(s)
        arr = np.array(scores, dtype=float)
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr.tolist()

class MHCLSynModel:
    def __init__(self, ds: str, root: str, alpha: float = 0.7):
        self.ds = ds
        self.root = root
        self.alpha = alpha
        self.pair_idx = {}
        self.S_atc = None
        self.S_tgt = None
        self.name_to_idx = {}

    def fit(self, train_items):
        for it in train_items:
            if int(it["label"]) != 1:
                continue
            a,b,c = it["drugA"], it["drugB"], it["drugC"]
            pairs = [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]
            for p in pairs:
                self.pair_idx[p] = self.pair_idx.get(p,0)+1
        if self.ds == 'breast':
            names, _ = load_incidence(os.path.join(self.root, 'Breast_141_Drug_Combination_Incidence_Matrix.csv'))
            names2, S_atc, S_tgt = build_breast(self.root)
        else:
            names, _ = load_incidence(os.path.join(self.root, 'Lung_155_Drug_Combination_Incidence_Matrix.xlsx'))
            names2, S_atc, S_tgt = build_lung(self.root)
        self.name_to_idx = {n:i for i,n in enumerate(names)}
        self.S_atc = S_atc
        self.S_tgt = S_tgt

    def score(self, items):
        scores = []
        for it in items:
            a,b,c = str(it["drugA"]), str(it["drugB"]), str(it["drugC"])
            pairs = [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]
            s_pair = sum(self.pair_idx.get(p,0) for p in pairs)
            ia = self.name_to_idx.get(a); ib = self.name_to_idx.get(b); ic = self.name_to_idx.get(c)
            if ia is not None and ib is not None and ic is not None:
                sims = [self.S_atc[ia, ib] + self.S_tgt[ia, ib], self.S_atc[ia, ic] + self.S_tgt[ia, ic], self.S_atc[ib, ic] + self.S_tgt[ib, ic]]
                s_sim = sum(sims)
            else:
                s_sim = 0.0
            s = self.alpha * s_pair + (1.0 - self.alpha) * s_sim
            scores.append(float(s))
        arr = np.array(scores, dtype=float)
        if arr.max() > 0:
            arr = arr / arr.max()
        return arr.tolist()

def build_model(name: str, ds: str, root: str):
    n = name.lower()
    if n == 'higsyn':
        return HIGSynRunner(root, ds)
    if n == 'mhclsyn':
        return MHCLSynModel(ds, root)
    if n in ["deepsynergy","gcn","gat"]:
        if n == 'deepsynergy':
            return DeepSynergyRunner(root, ds)
        if n.startswith('gcn'):
            # parse threshold suffix e.g., gcn_t005
            thresh = 0.1
            if '_t' in n:
                try:
                    val = n.split('_t')[1]
                    thresh = int(val)/1000.0 if len(val)==3 else float(val)
                except Exception:
                    pass
            return GCNRunner(root, ds, thresh=thresh)
        if n.startswith('gat'):
            thresh = 0.1
            if '_t' in n:
                try:
                    val = n.split('_t')[1]
                    thresh = int(val)/1000.0 if len(val)==3 else float(val)
                except Exception:
                    pass
            return GATRunner(root, ds, thresh=thresh)
        return PairTrainModel()
    return PairTrainModel()
