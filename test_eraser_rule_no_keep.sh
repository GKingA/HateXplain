cd eraserbenchmark

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/majority_all --results ../explanations_dicts/explanation_dict_all_bow_majority_all.json --score_file ../model_explain_output_all_bow_majority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/majority_all --results ../explanations_dicts/explanation_dict_rationale_bow_majority_all.json --score_file ../model_explain_output_rationale_bow_majority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/majority_all --results ../explanations_dicts/explanation_dict_all_graph_majority_all.json --score_file ../model_explain_output_all_graph_majority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/majority_all --results ../explanations_dicts/explanation_dict_feature_graph_majority_all.json --score_file ../model_explain_output_feature_graph_majority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/majority_all --results ../explanations_dicts/explanation_dict_rationale_graph_majority_all.json --score_file ../model_explain_output_rationale_graph_majority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/majority_all --results ../explanations_dicts/hand_rules_majority_all.json --score_file ../model_explain_output_hand_rule_majority_all.json


python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_all --results ../explanations_dicts/explanation_dict_all_bow_minority_all.json --score_file ../model_explain_output_all_bow_minority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_all --results ../explanations_dicts/explanation_dict_rationale_bow_minority_all.json --score_file ../model_explain_output_rationale_bow_minority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_all --results ../explanations_dicts/explanation_dict_all_graph_minority_all.json --score_file ../model_explain_output_all_graph_minority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_all --results ../explanations_dicts/explanation_dict_feature_graph_minority_all.json --score_file ../model_explain_output_feature_graph_minority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_all --results ../explanations_dicts/explanation_dict_rationale_graph_minority_all.json --score_file ../model_explain_output_rationale_graph_minority_all.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_all --results ../explanations_dicts/hand_rules_minority_all.json --score_file ../model_explain_output_hand_rule_minority_all.json


python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_pure --results ../explanations_dicts/explanation_dict_all_bow_minority_pure.json --score_file ../model_explain_output_all_bow_minority_pure.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_pure --results ../explanations_dicts/explanation_dict_rationale_bow_minority_pure.json --score_file ../model_explain_output_rationale_bow_minority_pure.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_pure --results ../explanations_dicts/explanation_dict_all_graph_minority_pure.json --score_file ../model_explain_output_all_graph_minority_pure.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_pure --results ../explanations_dicts/explanation_dict_feature_graph_minority_pure.json --score_file ../model_explain_output_feature_graph_minority_pure.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_pure --results ../explanations_dicts/explanation_dict_rationale_graph_minority_pure.json --score_file ../model_explain_output_rationale_graph_minority_pure.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/rules/minority_pure --results ../explanations_dicts/hand_rules_minority_pure.json --score_file ../model_explain_output_hand_rule_minority_pure.json
