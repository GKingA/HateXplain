import torch
import transformers
from transformers import *
import glob
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
from transformers import BertTokenizer

#### common utils
from Models.utils import fix_the_random, format_time, get_gpu, return_params

#### metric utils
from Models.utils import masked_cross_entropy, softmax, return_params

#### model utils
from Models.utils import save_normal_model, save_bert_model, load_model
from tqdm import tqdm
from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
import matplotlib.pyplot as plt
import time
import os
import GPUtil
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from Models.otherModels import *
from sklearn.preprocessing import LabelEncoder
from Preprocess.dataCollect import (
    get_test_data,
    convert_data,
    get_annotated_data,
    transform_dummy_data,
)
from TensorDataset.datsetSplitter import encodeData
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import ast
from torch.nn import LogSoftmax
from lime.lime_text import LimeTextExplainer
import numpy as np
import argparse
import GPUtil

# In[3]:


dict_data_folder = {
    "2": {"data_file": "Data/dataset.json", "class_label": "Data/classes_two.npy"},
    "3": {"data_file": "Data/dataset.json", "class_label": "Data/classes.npy"},
}

def select_model(params, embeddings):
    if params["bert_tokens"]:
        if params["what_bert"] == "weighted":
            model = SC_weighted_BERT.from_pretrained(
                params[
                    "path_files"
                ],  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=params["num_classes"],  # The number of output labels
                output_attentions=True,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
                hidden_dropout_prob=params["dropout_bert"],
                params=params,
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        text = params["model_name"]
        if text == "birnn":
            model = BiRNN(params, embeddings)
        elif text == "birnnatt":
            model = BiAtt_RNN(params, embeddings, return_att=True)
        elif text == "birnnscrat":
            model = BiAtt_RNN(params, embeddings, return_att=True)
        elif text == "cnn_gru":
            model = CNN_GRU(params, embeddings)
        elif text == "lstm_bad":
            model = LSTM_bad(params)
        else:
            print("Error in model name!!!!")
        return model


def standaloneEval_with_rational(
    params, test_data=None, extra_data_path=None, topk=2, keep=False, use_ext_df=False, use_test=False, negative_rationale=False
):
    #     device = torch.device("cpu")
    if torch.cuda.is_available() and params["device"] == "cuda":
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print("Since you dont want to use GPU, using the CPU instead.")
        device = torch.device("cpu")

    embeddings = None
    if params["bert_tokens"]:
        train, val, test = createDatasetSplit(params)
        vocab_own = None
        vocab_size = 0
        padding_idx = 0
    else:
        train, val, test, vocab_own = createDatasetSplit(params)
        params["embed_size"] = vocab_own.embeddings.shape[1]
        params["vocab_size"] = vocab_own.embeddings.shape[0]
        embeddings = vocab_own.embeddings
    if params["auto_weights"]:
        if use_test:
            y_test = [ele[2] for ele in test]
        else:
            y_test = [ele[2] for ele in val]
        encoder = LabelEncoder()
        encoder.classes_ = np.load(dict_data_folder[str(params["num_classes"])]["class_label"], allow_pickle=True)
        params["weights"] = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y_test), y=y_test
        ).astype("float32")
    if extra_data_path is not None:
        params_dash = {}
        params_dash["num_classes"] = params["num_classes"] # 3
        params_dash["data_file"] = extra_data_path
        params_dash["class_names"] = dict_data_folder[str(params["num_classes"])][
            "class_label"
        ]
        temp_read = get_annotated_data(params_dash)
        with open("Data/post_id_divisions.json", "r") as fp:
            post_id_dict = json.load(fp)
        partition = "test" if use_test else "val"
        if not keep:
            temp_read = temp_read[
                temp_read["post_id"].isin(post_id_dict[partition])
                & (temp_read["final_label"].isin(["hatespeech", "offensive", "toxic"]))
            ]
        else:
            temp_read = temp_read[
                temp_read["post_id"].isin(post_id_dict[partition])
            ]
        test_data = get_test_data(temp_read, params, message="text")
        test_extra = encodeData(test_data, vocab_own, params)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    elif use_ext_df:
        test_extra = encodeData(test_data, vocab_own, params)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    else:
        test_dataloader = combine_features(test, params, is_train=False)
    model = select_model(params, embeddings)
    if params["bert_tokens"] == False:
        model = load_model(model, params)
    if params["device"] == "cuda":
        model.cuda()
    model.eval()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables
    if (extra_data_path != None) or (use_ext_df == True):
        post_id_all = list(test_data["Post_id"])
    else:
        post_id_all = list(test["Post_id"])

    print("Running eval on test data...")
    t0 = time.time()
    true_labels = []
    pred_labels = []
    logits_all = []
    attention_all = []
    input_mask_all = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        # model.zero_grad()
        outputs = model(
            b_input_ids,
            attention_vals=b_att_val,
            attention_mask=b_input_mask,
            labels=None,
            device=device,
        )
        #         m = nn.Softmax(dim=1)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()

        if params["bert_tokens"]:
            attention_vectors = np.mean(
                outputs[1][11][:, :, 0, :].detach().cpu().numpy(), axis=1
            )
        else:
            attention_vectors = outputs[1].detach().cpu().numpy()

        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)
        attention_all += list(attention_vectors)
        input_mask_all += list(batch[2].detach().cpu().numpy())

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    if use_ext_df == False:
        testf1 = f1_score(true_labels, pred_labels, average="macro")
        testacc = accuracy_score(true_labels, pred_labels)
        # testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testprecision = precision_score(true_labels, pred_labels, average="macro")
        testrecall = recall_score(true_labels, pred_labels, average="macro")

        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.3f}".format(testacc))
        print(" Fscore: {0:.3f}".format(testf1))
        print(" Precision: {0:.3f}".format(testprecision))
        print(" Recall: {0:.3f}".format(testrecall))
        # print(" Roc Auc: {0:.3f}".format(testrocauc))
        print(" Test took: {:}".format(format_time(time.time() - t0)))

    attention_vector_final = []
    for x, y in zip(attention_all, input_mask_all):
        temp = []
        for x_ele, y_ele in zip(x, y):
            if y_ele == 1:
                temp.append(x_ele)
        attention_vector_final.append(temp)

    list_dict = []

    for post_id, attention, logits, pred, ground_truth in zip(
        post_id_all, attention_vector_final, logits_all_final, pred_labels, true_labels
    ):
        #         if(ground_truth==1):
        #             continue
        temp = {}
        encoder = LabelEncoder()
        encoder.classes_ = np.load(dict_data_folder[str(params["num_classes"])]["class_label"], allow_pickle=True)
        pred_label = encoder.inverse_transform([pred])[0]
        ground_label = encoder.inverse_transform([ground_truth])[0]
        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        if params["num_classes"] == 3:
            temp["classification_scores"] = {
                "hatespeech": logits[0],
                "normal": logits[1],
                "offensive": logits[2],
            }
        else:
            temp["classification_scores"] = {
                 "toxic": logits[1],
                 "non-toxic": logits[0]
            }

        topk_indicies = sorted(range(len(attention)), key=lambda i: attention[i])[
            -topk:
        ]

        temp_hard_rationales = []
        for ind in topk_indicies:
            temp_hard_rationales.append({"end_token": ind + 1, "start_token": ind})
        if not negative_rationale or (negative_rationale and pred_label in ["toxic", "hatespeech", "offensive"]):
            temp["rationales"] = [
                {
                    "docid": post_id,
                    "hard_rationale_predictions": temp_hard_rationales,
                    "soft_rationale_predictions": attention,
                    # "soft_sentence_predictions":[1.0],
                    "truth": ground_truth,
                }
            ]
        else:
            temp["rationales"] = [
                {
                    "docid": post_id,
                    "hard_rationale_predictions": [],
                    "soft_rationale_predictions": [0 for _ in attention],
                    "truth": ground_truth,
                }
            ]
        list_dict.append(temp)

    return list_dict, test_data


# In[115]:


def get_final_dict_with_rational(params, test_data=None, topk=5, keep=False, test=False, negative_rationale=False):
    list_dict_org, test_data = standaloneEval_with_rational(
        params, extra_data_path=test_data, topk=topk, keep=keep, use_test=test, negative_rationale=negative_rationale
    )
    test_data_with_rational = convert_data(
        test_data, params, list_dict_org, rational_present=True, topk=topk
    )
    list_dict_with_rational, _ = standaloneEval_with_rational(
        params, test_data=test_data_with_rational, topk=topk, keep=keep, use_ext_df=True, use_test=test, negative_rationale=negative_rationale
    )
    test_data_without_rational = convert_data(
        test_data, params, list_dict_org, rational_present=False, topk=topk
    )
    list_dict_without_rational, _ = standaloneEval_with_rational(
        params, test_data=test_data_without_rational, topk=topk, keep=keep, use_ext_df=True, use_test=test, negative_rationale=negative_rationale
    )
    final_list_dict = []
    for ele1, ele2, ele3 in zip(
        list_dict_org, list_dict_with_rational, list_dict_without_rational
    ):
        ele1["sufficiency_classification_scores"] = ele2["classification_scores"]
        ele1["comprehensiveness_classification_scores"] = ele3["classification_scores"]
        final_list_dict.append(ele1)
    return final_list_dict


# In[ ]:


# In[88]:


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description="Which model to use")

    # Add the arguments
    my_parser.add_argument(
        "model_to_use",
        metavar="--model_to_use",
        type=str,
        help="model config to use for evaluation",
    )

    my_parser.add_argument(
        "attention_lambda",
        metavar="--attention_lambda",
        type=str,
        help="required to assign the contribution of the atention loss",
    )
    my_parser.add_argument(
        "--keep_neutral",
        "-kn",
        action="store_true",
        default=False,
        help="keep neutral parts of the dataset",
    )
    my_parser.add_argument(
        "--bert_mask",
        "-bm",
        action="store_true",
        default=False,
        help="replace the rationale/not rationale with mask token instead of deleting it"
    )
    my_parser.add_argument(
        "--test_data",
        "-td",
        type=str,
        help="data to use for evaluation, defaults to the test portion of the data inn the json file"
    )
    my_parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="whether to run the test on the test portion of the dataset. Default is validation."
    )
    my_parser.add_argument(
        "--negative_rationale",
        "-nr",
        action="store_true",
        help="Whether to leave out the negative rationale's prediction or keep them. Default keeps them."
    )

    args = my_parser.parse_args()

    model_to_use = args.model_to_use

    params = return_params(
        model_to_use, float(args.attention_lambda)
    )
    # params["variance"] = 1
    # params["num_classes"] = 3  use the one from the config file
    # params["device"] = "cpu"
    fix_the_random(seed_val=params["random_seed"])
    params["class_names"] = dict_data_folder[str(params["num_classes"])]["class_label"]
    params["data_file"] = dict_data_folder[str(params["num_classes"])]["data_file"] if "data_file" not in params else params["data_file"]
    if args.test_data is not None:
        data_file = args.test_data
    else:
        data_file = params["data_file"]
    # test_data=get_test_data(temp_read,params,message='text')
    final_list_dict = get_final_dict_with_rational(params, data_file, topk=5, keep=args.keep_neutral, test=args.test, negative_rationale=args.negative_rationale)

    path_name = model_to_use
    additive = "_" if not args.negative_rationale else "_not_neg_"
    if args.test_data is None:
        path_name_explanation = (
            "explanations_dicts/"
            + path_name.split("/")[1].split(".")[0]
            + "_"
            + str(params["att_lambda"])
            + additive
            + "explanation_top5.json"
        )
    else:
        path_name_explanation = (
                "explanations_dicts/"
                + path_name.split("/")[1].split(".")[0]
                + "_"
                + str(params["att_lambda"])
                + additive
                + "explanation_top5_"
                + os.path.basename(data_file).split(".")[0]
        )

        if args.bert_mask:
            path_name_explanation += "_bert_mask.json"
        else:
            path_name_explanation += ".json"
    os.makedirs(os.path.dirname(path_name_explanation), exist_ok=True)
    with open(path_name_explanation, "w") as fp:
        fp.write("\n".join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))

# In[ ]:
