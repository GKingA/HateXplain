import json
import os

import pandas as pd


def read_json(path):
    with open(path) as cs:
        classification_scores = json.load(cs)
        print(f'\n{os.path.basename(path)}\n')
        print(f'Toxic F1\t{classification_scores["classification_scores"]["prf"]["toxic"]["f1-score"]}')
        print(f'Sufficiency\t{classification_scores["classification_scores"]["sufficiency"]}')
        print(f'Comprehensiveness\t{classification_scores["classification_scores"]["comprehensiveness"]}')
        print(f'IOU F1\t{classification_scores["iou_scores"][0]["macro"]["f1"]}')
        print(f'Token F1\t{classification_scores["token_prf"]["instance_macro"]["f1"]}')
        print(f'AUPRC\t{classification_scores["token_soft_metrics"]["auprc"]}')


def scores(folders, mixmatch=False):
    majority = {}
    minority = {}
    pure = {}
    all_on_pure = {}
    pure_on_all = {}
    for folder in folders:
        paths = sorted(os.listdir(folder))
        for p in paths:
            if not p.endswith("json"):
                continue
            with open(os.path.join(folder, p)) as explanation:
                if not mixmatch:
                    if "majority" in p and "all" in p:
                        majority[p] = json.load(explanation)
                    elif "minority" in p and "all" in p:
                        minority[p] = json.load(explanation)
                    elif "minority" in p and "pure" in p:
                        pure[p] = json.load(explanation)
                else:
                    if p.index("all") < p.index("pure"):
                        all_on_pure[p] = json.load(explanation)
                    else:
                        pure_on_all[p] = json.load(explanation)
    if not mixmatch:
        return majority, minority, pure
    else:
        return all_on_pure, pure_on_all


def explainability(dictionary):
    not_names = ["model", "explain", "output", "majority", "minority", "all", "pure"]
    quant = {}
    qual = {}
    for filename, score in dictionary.items():
        name = " ".join([e.capitalize() for e in filename.split(".")[0].split("_") if e not in not_names])
        quant[name] = {}
        qual[name] = {}
        quant[name]["Model"] = name
        qual[name]["Model"] = name
        quant[name]["IOU F1"] = score["iou_scores"][0]["macro"]["f1"] * 100
        quant[name]["Token F1"] = score["token_prf"]["instance_macro"]["f1"] * 100
        qual[name]["Suff."] = score["classification_scores"]["sufficiency"]
        qual[name]["Comp."] = score["classification_scores"]["comprehensiveness"]
    df = pd.read_json(json.dumps(quant)).T
    df_qual = pd.read_json(json.dumps(qual)).T
    return df.to_latex(index = False, float_format="{:.2f}%".format), df_qual.to_latex(index = False, float_format="{:.2f}".format)


def performance(dictionary):
    not_names = ["model", "explain", "output", "majority", "minority", "all", "pure", "lime", "rationale"]
    quant = {}
    for filename, score in dictionary.items():
        name = " ".join([e.capitalize() for e in filename.split(".")[0].split("_") if e not in not_names])
        quant[name] = score["classification_scores"]["prf"]["toxic"]
        quant[name]["Model"] = name
    df = pd.read_json(json.dumps(quant)).T
    df = df[["Model", "precision", "recall", "f1-score"]]
    df.precision = df.precision * 100
    df.recall = df.recall * 100
    df["f1-score"] = df["f1-score"] * 100
    return df.to_latex(index=False, float_format="{:.2f}%".format)


if __name__ == "__main__":
    folders = ["fair_eval"]
    maj, mino, pur = scores(folders)
    for i in [maj, mino, pur]:
        print(performance(i))
        qualitative_s = explainability(i)
        print(qualitative_s[0])
        print(qualitative_s[1])
