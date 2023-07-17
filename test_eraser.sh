cd eraserbenchmark

#lime

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/majority_all --results ../explanations_dicts/bestModel_birnn_2_class_women_majority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_birnn_majority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_all --results ../explanations_dicts/bestModel_birnn_2_class_women_minority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_birnn_minority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_pure --results ../explanations_dicts/bestModel_birnn_2_class_women_minority_pure_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_birnn_minority_pure_lime.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/majority_all --results ../explanations_dicts/bestModel_birnnatt_2_class_women_majority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_birnnatt_majority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_all --results ../explanations_dicts/bestModel_birnnatt_2_class_women_minority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_birnnatt_minority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_pure --results ../explanations_dicts/bestModel_birnnatt_2_class_women_minority_pure_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_birnnatt_minority_pure_lime.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/majority_all --results ../explanations_dicts/bestModel_birnnscrat_2_class_women_majority_all_explanation_with_lime_100_100.json --score_file ../model_explain_output_birnnscrat_majority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_all --results ../explanations_dicts/bestModel_birnnscrat_2_class_women_minority_all_explanation_with_lime_100_100.json --score_file ../model_explain_output_birnnscrat_minority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_pure --results ../explanations_dicts/bestModel_birnnscrat_2_class_women_minority_pure_explanation_with_lime_100_100.json --score_file ../model_explain_output_birnnscrat_minority_pure_lime.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/majority_all --results ../explanations_dicts/bestModel_cnn_gru_2_class_women_majority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_cnn_gru_majority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_all --results ../explanations_dicts/bestModel_cnn_gru_2_class_women_minority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_cnn_gru_minority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_pure --results ../explanations_dicts/bestModel_cnn_gru_2_class_women_minority_pure_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_cnn_gru_minority_pure_lime.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_majority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_2_class_women_majority_all_explanation_with_lime_100_0.001.json --score_file ../model_explain_output_bert_true_majority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_2_class_women_minority_all_explanation_with_lime_100_0.001.json --score_file ../model_explain_output_bert_true_minority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_pure --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_2_class_women_minority_pure_explanation_with_lime_100_0.001.json --score_file ../model_explain_output_bert_true_minority_pure_lime.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_majority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_FALSE_2_class_women_majority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_bert_false_majority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_FALSE_2_class_women_minority_all_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_bert_false_minority_all_lime.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_pure --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_FALSE_2_class_women_minority_pure_explanation_with_lime_100_1.0.json --score_file ../model_explain_output_bert_false_minority_pure_lime.json


#rationale

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/majority_all --results ../explanations_dicts/bestModel_birnnatt_2_class_women_majority_all_1.0_explanation_top5.json --score_file ../model_explain_output_birnnatt_majority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_all --results ../explanations_dicts/bestModel_birnnatt_2_class_women_minority_all_1.0_explanation_top5.json --score_file ../model_explain_output_birnnatt_minority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_pure --results ../explanations_dicts/bestModel_birnnatt_2_class_women_minority_pure_1.0_explanation_top5.json --score_file ../model_explain_output_birnnatt_minority_pure_rationale.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/majority_all --results ../explanations_dicts/bestModel_birnnscrat_2_class_women_majority_all_100_explanation_top5.json --score_file ../model_explain_output_birnnscrat_majority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_all --results ../explanations_dicts/bestModel_birnnscrat_2_class_women_minority_all_100_explanation_top5.json --score_file ../model_explain_output_birnnscrat_minority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/minority_pure --results ../explanations_dicts/bestModel_birnnscrat_2_class_women_minority_pure_100_explanation_top5.json --score_file ../model_explain_output_birnnscrat_minority_pure_rationale.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_majority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_2_class_women_majority_all_0.001_explanation_top5.json --score_file ../model_explain_output_bert_true_majority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_2_class_women_minority_all_0.001_explanation_top5.json --score_file ../model_explain_output_bert_true_minority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_pure --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_TRUE_2_class_women_minority_pure_0.001_explanation_top5.json --score_file ../model_explain_output_bert_true_minority_pure_rationale.json

python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_majority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_FALSE_2_class_women_majority_all_1.0_explanation_top5.json --score_file ../model_explain_output_bert_false_majority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_all --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_FALSE_2_class_women_minority_all_1.0_explanation_top5.json --score_file ../model_explain_output_bert_false_minority_all_rationale.json
python3 rationale_benchmark/metrics.py --split val --strict --data_dir ../Data/Evaluation/Model_Eval/bert_minority_pure --results ../explanations_dicts/bestModel_bert_base_uncased_Attn_train_FALSE_2_class_women_minority_pure_1.0_explanation_top5.json --score_file ../model_explain_output_bert_false_minority_pure_rationale.json

