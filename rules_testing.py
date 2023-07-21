import os.path

import pandas as pd
import json
import networkx as nx
from ast import literal_eval
from typing import List, Tuple
from argparse import ArgumentParser

from nltk import WordNetLemmatizer

from xpotato.graph_extractor.extract import FeatureEvaluator
from xpotato.dataset.explainable_dataset import ExplainableDataset
from tuw_nlp.graph.utils import graph_to_pn


def rationale_graph(df: pd.DataFrame):
    graphs = []
    rats = df[df.rationale == 1]
    rat_ids = [
        (literal_eval(r.rationale_lemma), r.graph)
        for (i, r) in rats.iterrows()
        if literal_eval(r.rationale_lemma) != []
    ]
    for ids, graph in rat_ids:
        g = graph.subgraph([i + 1 for i in ids])
        connected_components = list(nx.weakly_connected_components(g))
        graphs += [graph_to_pn(g.subgraph(c)) for c in connected_components]
    return graphs


def classification_score(match: pd.Series, target: str):
    classification = (
        "toxic" if target.capitalize() in match["Predicted label"] else "non-toxic"
    )
    not_class = "non-toxic" if classification == "toxic" else "toxic"
    return classification, not_class


def bow_model(
    potato_path: str,
    json_path: str,
    post_id: str,
    features: str,
    target: str,
    output_file: str,
    test: bool = False,
    delete_negative: bool=False,
) -> None:
    with open(features, "r") as feat_file:
        features = json.load(feat_file)
    with open(post_id, "r") as post_id_file:
        post_id_division = json.load(post_id_file)
    df = ExplainableDataset(
        path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}
    ).to_dataframe()
    og = pd.read_json(json_path).T
    port = "test" if test else "val"
    test_posts = og[og.post_id.isin(post_id_division[port])]
    words = features[target.capitalize()]
    lemmatizer = WordNetLemmatizer()
    with open(output_file, "w") as out:
        for (_, original), (_, potato) in zip(test_posts.iterrows(), df.iterrows()):
            if delete_negative and (potato.label != target.capitalize()):
                continue
            lemmata = [lemmatizer.lemmatize(t) for t in original["post_tokens"]]
            rats = [(lemma in words) * 1 for lemma in lemmata]
            if sum(rats) > 0:
                classification = "toxic"
                classification_scores = {"toxic": 1, "non-toxic": 0}
            else:
                classification = "non-toxic"
                classification_scores = {"toxic": 0, "non-toxic": 1}
            # Just for fairness
            wo = [lemma for lemma in lemmata if lemma not in words]
            wo_rats = [(lemma in words) * 1 for lemma in wo]
            if sum(wo_rats) > 0:
                wo_classification_scores = {"toxic": 1, "non-toxic": 0}
            else:
                wo_classification_scores = {"toxic": 0, "non-toxic": 1}
            just = [lemma for lemma in lemmata if lemma in words]
            just_rats = [(lemma in words) * 1 for lemma in just]
            if sum(just_rats) > 0:
                just_classification_scores = {"toxic": 1, "non-toxic": 0}
            else:
                just_classification_scores = {"toxic": 0, "non-toxic": 1}
            dictionary = {
                "annotation_id": original.post_id,
                "classification": classification,
                "classification_scores": classification_scores,
                "rationales": [
                    {
                        "docid": original.post_id,
                        "hard_rationale_predictions": [
                            {"end_token": index + 1, "start_token": index}
                            for index, i in enumerate(rats) if i == 1
                        ],
                        "soft_rationale_predictions": rats,
                        "truth": 1,
                    }
                ],
                "sufficiency_classification_scores": just_classification_scores,
                "comprehensiveness_classification_scores": wo_classification_scores,
            }
            out.write(f"{json.dumps(dictionary)}\n")


def graph_model(
    potato_path: str,
    json_path: str,
    post_id: str,
    features: str,
    target: str,
    output_file: str,
    test: bool = False,
    delete_negative: bool=False,
) -> Tuple[List[List[int]], List[str]]:
    df = ExplainableDataset(
        path=potato_path, label_vocab={"None": 0, target.capitalize(): 1}
    ).to_dataframe()
    with open(features, "r") as feat_file:
        features = json.load(feat_file)
    with open(post_id, "r") as post_id_file:
        post_id_division = json.load(post_id_file)
    og = pd.read_json(json_path).T
    port = "test" if test else "val"
    test_posts = og[og.post_id.isin(post_id_division[port])]
    print(len(test_posts), len(df))
    evaluator = FeatureEvaluator()
    feat_columns = []
    feats = []
    match = evaluator.match_features(
        df,
        features[target.capitalize()],
        multi=True,
        return_subgraphs=True,
        allow_multi_graph=True,
    )
    joined = df.join(match)
    removed = []
    only_rat = []
    for subs, graph in zip(joined["Matched subgraph"], joined.graph):
        rat = nx.DiGraph()
        s_graph = graph.copy()
        for rule_sub in subs:
            for sub in rule_sub:
                s_graph.remove_nodes_from(sub.nodes)
                try:
                    rat = nx.union(rat, sub)
                except nx.exception.NetworkXError:
                    rat = nx.disjoint_union(rat, sub)
        removed.append(s_graph)
        only_rat.append(rat)
    just_rationale = joined.copy()
    just_rationale.graph = only_rat
    just_match = evaluator.match_features(
        just_rationale,
        features[target.capitalize()],
        multi=True,
        return_subgraphs=True,
        allow_multi_graph=True,
    )  # Sufficiency
    without_rationale = joined.copy()
    without_rationale.graph = removed
    without_match = evaluator.match_features(
        without_rationale,
        features[target.capitalize()],
        multi=True,
        return_subgraphs=True,
        allow_multi_graph=True,
    )  # Comprehension
    with open(output_file, "w") as out:
        for (_, m), (_, just), (_, wo), (_, original), (_, og) in zip(
            match.iterrows(),
            just_match.iterrows(),
            without_match.iterrows(),
            test_posts.iterrows(),
            df.iterrows(),
        ):
            if delete_negative and (og.label != target.capitalize()):
                continue
            classification, not_class = classification_score(m, target)
            just_classification, just_not_class = classification_score(just, target)
            wo_classification, wo_not_class = classification_score(wo, target)
            if classification == "toxic" and wo_classification == "toxic":
                breakpoint()
            rationale_ids = [
                n
                for subgraph in m["Matched subgraph"]
                for rule_graph in subgraph
                for n in rule_graph.nodes()
            ]
            # Correct issues with UD parser, that creates new nodes for < and >
            issues = [
                n[0]
                for n in og.graph.nodes(data=True)
                if n[1]["name"] == "LT" or n[1]["name"] == "GT"
            ]
            corrected_rationale_ids = []
            for rat in rationale_ids:
                n = 0
                for issue in issues:
                    if issue < rat:
                        n += 1
                corrected_rationale_ids.append(rat - n)
            dictionary = {
                "annotation_id": original.post_id,
                "classification": classification,
                "classification_scores": {classification: 1, not_class: 0},
                "rationales": [
                    {
                        "docid": original.post_id,
                        "hard_rationale_predictions": [
                            {"end_token": i, "start_token": i - 1}
                            for i in corrected_rationale_ids
                        ],
                        "soft_rationale_predictions": [
                            1 if i + 1 in corrected_rationale_ids else 0
                            for i, _ in enumerate(original.post_tokens)
                        ],
                        "truth": 1,
                    }
                ],
                "sufficiency_classification_scores": {
                    just_classification: 1,
                    just_not_class: 0,
                },
                "comprehensiveness_classification_scores": {
                    wo_classification: 1,
                    wo_not_class: 0,
                },
            }
            out.write(f"{json.dumps(dictionary)}\n")

    for feature in features[target.capitalize()]:
        match = evaluator.match_features(
            df, [feature], multi=True, return_subgraphs=True, allow_multi_graph=True
        )
        feats.append(str(feature))
        feat_columns.append(
            [
                (r["Predicted label"] == target.capitalize()) * 1
                for (_, r) in match.iterrows()
            ]
        )
    feature_vectors = list(map(list, zip(*feat_columns)))
    return feature_vectors, feats


if __name__ == "__main__":
    parse_args = ArgumentParser()
    parse_args.add_argument("--hand_written", "-hw", help="path to the hand written rules, if not given, "
                                                          "the script will run on generated rules")
    parse_args.add_argument("--base_path", "-p", help="path to the potato file (dirpath, or exact path, if dirpath, "
                                                      "the assumed format is [voting]_[test or val]_[filter].tsv)")
    parse_args.add_argument("--base_original", "-og", help="path to the original file (dirpath, or exact path, if dirpath, "
                                                          "the assumed format is [voting]_[filter].json)")
    parse_args.add_argument("--features", "-fe", help="path to the feature file")
    parse_args.add_argument("--post_id", "-i", help="path to the post_id_divisions.json file")
    parse_args.add_argument("--test", "-t", action="store_true", help="run it on test data instead of validation data.")
    parse_args.add_argument("--voting", "-v", choices=["majority", "minority", "maj", "min"], help="the voting used for the data labeling.")
    parse_args.add_argument("--filter", "-fi", choices=["all", "pure", "a", "p"], help="the filtering used for the data.")
    parse_args.add_argument("--method", "-m", choices=["rationale_graph", "feature_graph", "all_graph", "rationale_bow", "all_bow"])
    parse_args.add_argument("--target", "-tar", help="the target of hate.", default="Women")
    parse_args.add_argument("--delete_negative", "-dn", action="store_true", help="delete the ground truth negative elements.")
    parse_args.add_argument("--output", "-o", help="the output file of the prediction, "
                                                   "default is explanation_dict_[method]_[voting]_[filter].json.")
    args = parse_args.parse_args()
    num = {("rationale_graph", "majority", "all"): 7,
           ("rationale_graph", "minority", "all"): 14,
           ("rationale_graph", "minority", "pure"): 5,
           ("feature_graph", "majority", "all"): 6,
           ("feature_graph", "minority", "all"): 13,
           ("feature_graph", "minority", "pure"): 4,
           ("all_graph", "majority", "all"): 7,
           ("all_graph", "minority", "all"): 14,
           ("all_graph", "minority", "pure"): 5,
           ("rationale_bow", "majority", "all"): 3,
           ("rationale_bow", "minority", "all"): 5,
           ("rationale_bow", "minority", "pure"): 4,
           ("all_bow", "majority", "all"): 3,
           ("all_bow", "minority", "all"): 2,
           ("all_bow", "minority", "pure"): 3,
           }
    if args.hand_written is None:
        types = ["rationale_graph", "feature_graph", "all_graph", "rationale_bow", "all_bow"]
    else:
        types = [args.hand_written]
    for type_name in types:
        for voting in ["majority", "minority"]:
            for filtering in ["all", "pure"]:
                if voting == "majority" and filtering == "pure":
                    continue
                if args.hand_written is not None:
                    args = parse_args.parse_args("-p women/m/ -og women/m/ "
                                                 f"-fe {type_name} "
                                                 "-i post_id_divisions.json "
                                                 f"-v {voting} -fi {filtering} -dn "
                                                 f"-o hand_rules_{voting}_{filtering}.json".split(" "))
                else:
                    number = num[(type_name, voting, filtering)]
                    args = parse_args.parse_args("-p women/m/ -og women/m/ "
                                                 f"-fe {type_name}_features/{type_name}_{voting}_{filtering}_{number}.json "
                                                 "-i post_id_divisions.json "
                                                 f"-v {voting} -fi {filtering} -m {type_name} -dn".split(" "))

                voting = args.voting
                filter = args.filter
                val = "val" if not args.test else "test"
                if os.path.isdir(args.base_path):
                    validation_file = os.path.join(args.base_path, f"{voting}_{val}_{filter}.tsv")
                else:
                    validation_file = args.base_path

                if os.path.isdir(args.base_path):
                    original_file = os.path.join(args.base_original, f"{voting}_{filter}.json")
                else:
                    original_file = args.base_original

                if args.output is None:
                    output = f"explanation_dict_{args.method}_{voting}_{filter}.json"
                else:
                    output = args.output

                in_feats = args.features
                post_id_divisions = args.post_id
                target = args.target

                if hand or "graph" in args.method:
                    feat, names = graph_model(
                        validation_file,
                        original_file,
                        post_id_divisions,
                        in_feats,
                        target,
                        output,
                        delete_negative=args.delete_negative,
                    )
                else:
                    bow_model(
                        validation_file,
                        original_file,
                        post_id_divisions,
                        in_feats,
                        target,
                        output,
                        delete_negative=args.delete_negative,
                    )
