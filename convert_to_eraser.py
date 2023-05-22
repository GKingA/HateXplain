import json
import os
import more_itertools as mit
from tqdm import tqdm
from Preprocess import *
from Preprocess.dataCollect import *
from argparse import ArgumentParser


dict_data_folder={
      '2':{'data_file':'Data/dataset.json','class_label':'Data/classes_two.npy'},
      '3':{'data_file':'Data/dataset.json','class_label':'Data/classes.npy'}
}

# Load the whole dataset and get the tokenwise rationales
def get_training_data(data, params_data):

    if params_data['bert_tokens']:
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    else:
        print('Loading Normal tokenizer...')
        tokenizer = None

    final_binny_output = []
    print('total_data', len(data))
    for index, row in tqdm(data.iterrows(), total=len(data)):
        annotation = row['final_label']
        post_id = row['post_id']
        annotation_list = [row['label1'], row['label2'], row['label3']]

        if annotation != 'undecided':
            tokens_all, attention_masks = returnMask(row, params_data, tokenizer)
            final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])

    return final_binny_output

# https://stackoverflow.com/questions/2154249/identify-groups-of-continuous-numbers-in-a-list
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each == 1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each) == int:
            start = each
            end = each + 1
        elif len(each) == 2:
            start = each[0]
            end = each[1] + 1
        else:
            print('error')

        output.append({"docid": post_id,
                       "end_sentence": -1,
                       "end_token": end,
                       "start_sentence": -1,
                       "start_token": start,
                       "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrices defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):
    final_output = []

    if save_split:
        train_fp = open(save_path + 'train.jsonl', 'w')
        val_fp = open(save_path + 'val.jsonl', 'w')
        test_fp = open(save_path + 'test.jsonl', 'w')

    for tcount, eachrow in enumerate(dataset):

        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        if majority_label == 'normal':
            continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        # For this work, we have considered the union of explanations. Other options could be explored as well.
        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]

        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            if not os.path.exists(save_path + 'docs'):
                os.makedirs(save_path + 'docs')

            with open(save_path + 'docs/' + post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp) + '\n')
            else:
                print(post_id)

    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()

    return final_output

# All is based on Explainability_Calculation_NB.ipynb
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--params", "-p", help="Path to the config file")
    arg_parser.add_argument("--bert", "-b", help="Use bert tokenizer", action="strore_true", default=False)
    arg_parser.add_argument("--save_path", "-s", help="Where to save the output", default=os.path.join(os.path.dirname(__file__), "Data/Evaluation/Model_Eval/"))
    args = arg_parser.parse_args()

    method = 'union'
    save_split = True
    save_path = args.save_path  # The dataset in Eraser Format will be stored here.
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "/Data/post_id_divisions.json")) as fp:
        id_division = json.load(fp)

    with open(args.params) as fp:
        params = json.load(fp)

    params["class_names"] = dict_data_folder[str(int(params['num_classes']))]['class_label']

    params['include_special'] = False
    params['bert_tokens'] = args.bert  # False/True
    params['type_attention'] = 'softmax'
    params['set_decay'] = 0.1
    params['majority'] = 2
    params['max_length'] = 128
    params['variance'] = 10
    params['window'] = 4
    params['alpha'] = 0.5
    params['p_value'] = 0.8
    params['method'] = 'additive'
    params['decay'] = False
    params['normalized'] = False
    params['not_recollect'] = True

    data_all_labelled = get_annotated_data(params)
    training_data = get_training_data(data_all_labelled, params)

    convert_to_eraser_format(training_data, method, save_split, save_path, id_division)
