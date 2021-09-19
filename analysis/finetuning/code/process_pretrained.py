import torch
import os, sys, argparse
import pandas as pd
import json

# All the MaskedLM
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import SqueezeBertTokenizer, SqueezeBertForMaskedLM
from transformers import MobileBertTokenizer, MobileBertForMaskedLM
from transformers import LongformerTokenizer, LongformerForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM

# import from maskedlm predict_prefix_probability to do the performance analysis
from maskedlm import predict_prefix_probability, one_mask_generation

MODELS = [
    "bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base",\
    "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",\
    "allenai/longformer-large-4096", "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2", "distilbert-base-cased"
]

adversarial_registry_path = os.getcwd() + "/../../data/process/associativebias_registry.tsv"
ASSOCIATIVE_BIAS_REGISTRY = pd.read_csv(adversarial_registry_path, sep="\t")


###### HELPERS ######

def parse_args():
    parser = argparse.ArgumentParser(description='finetuning BERT')
    parser.add_argument('-m', help="model name. {}".format(MODELS), default='bert-base-cased')
    parser.add_argument('-afmode', help="adversarial filtering or not ['none', 'model_specific', 'all_models']",\
                                                    default='model_specific')
    # finetuning args
    parser.add_argument('-dept', help="whether we're on the department machine or not", default="True")
    parser.add_argument('-overwrite', help="overwriting previous resukts or not", default="False")
    return parser.parse_args()

#########################


def get_pretrained_path_template_and_model(model_name):
    assert model_name in MODELS
    name_to_tokenizer_model = {
        "bert-base-cased": (BertTokenizer, BertForMaskedLM),
        "bert-large-cased-whole-word-masking": (BertTokenizer, BertForMaskedLM),
        "roberta-base": (RobertaTokenizer, RobertaForMaskedLM),
        "roberta-large": (RobertaTokenizer, RobertaForMaskedLM),
        "distilroberta-base": (RobertaTokenizer, RobertaForMaskedLM),
        "squeezebert/squeezebert-uncased": (SqueezeBertTokenizer, SqueezeBertForMaskedLM), 
        "google/mobilebert-uncased": (MobileBertTokenizer, MobileBertForMaskedLM),
        "allenai/longformer-base-4096": (LongformerTokenizer, LongformerForMaskedLM),
        "allenai/longformer-large-4096": (LongformerTokenizer, LongformerForMaskedLM),
        "albert-base-v2": (AlbertTokenizer, AlbertForMaskedLM), 
        "albert-large-v2": (AlbertTokenizer, AlbertForMaskedLM),
        "albert-xlarge-v2": (AlbertTokenizer, AlbertForMaskedLM),
        "albert-xxlarge-v2": (AlbertTokenizer, AlbertForMaskedLM),
        "distilbert-base-cased": (DistilBertTokenizer, DistilBertForMaskedLM)
    }
    tokenizer_module, model_module = name_to_tokenizer_model[model_name]
    return model_module.from_pretrained(model_name).eval(), tokenizer_module.from_pretrained(model_name)

# Done on dept: albert-base-v2 (d), google/mobilebert-uncased, roberta-base (d), squeezebert/squeezebert-uncased (d),
#               distilroberta-base (d)
# Hung: roberta-large, allenai/longformer-base-4096, allenai/longformer-large-4096

def get_three_tiers_dataset(model_name):
    ### HELPERS ###
    def populate_set(row, to_populate_set):
        to_populate_set.add((row["Word"], row["Associative Bias"], row["Alternative"]))

    def in_set(row, populated_set):
        return (row["Word"], row["Associative Bias"], row["Alternative"]) in populated_set

    def pass_all_associative_bias_tests(row):
        return_bool = True
        for m in MODELS:
            return_bool = return_bool and row[m]
        return return_bool

    ###############
    # first, get the challenge
    challenge_path = os.getcwd() + "/../../data/final/winoventi_bert_large_final.tsv"
    original_challenges = pd.read_csv(challenge_path, sep="\t")
    # now onto the part where we construct model_specific_challenges
    # filter challenge based on it being in the adversarial ones or not for this model
    # this part is for
    FILTERED_ASSOCIATIVE_BIAS_REGISTRY = ASSOCIATIVE_BIAS_REGISTRY[ASSOCIATIVE_BIAS_REGISTRY[model_name]]
    passed = set()
    FILTERED_ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: populate_set(x, passed), axis=1)
    model_specific_challenges = original_challenges[original_challenges.apply(lambda x: in_set(x, passed), axis=1)]
    # now onto the part where we construct all_models_challenges
    FILTERED_ASSOCIATIVE_BIAS_REGISTRY = ASSOCIATIVE_BIAS_REGISTRY[ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: pass_all_associative_bias_tests(x), \
                                                                                                                        axis=1)]
    passed = set()
    FILTERED_ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: populate_set(x, passed), axis=1)
    all_models_challenges = original_challenges[original_challenges.apply(lambda x: in_set(x, passed), axis=1)]
    return original_challenges, model_specific_challenges, all_models_challenges


def get_accuracies_one_two(model, tokenizer, model_name, dataset):
    ##### HELPER FUNCTIONS #####
    def get_probability(masked_prompt, word):
        if "longformer" in model_name.lower():
            alt_word = f"Ġ{word}"
            return max(
                predict_prefix_probability(tokenizer, model, masked_prompt, word, masked_prefix=masked_prompt),
                predict_prefix_probability(tokenizer, model, masked_prompt, alt_word, masked_prefix=masked_prompt)
            )
        if "roberta" in model_name.lower():
            alt_word = f"Ġ{word}"
            return max(
                predict_prefix_probability(tokenizer, model, masked_prompt, word, masked_prefix=masked_prompt),
                predict_prefix_probability(tokenizer, model, masked_prompt, alt_word, masked_prefix=masked_prompt)
            )
        if "albert" in model_name.lower():
            alt_word = "▁{}".format(word)
            return max(predict_prefix_probability(tokenizer, model, masked_prompt, word, masked_prefix=masked_prompt), 
                        predict_prefix_probability(tokenizer, model, masked_prompt, alt_word, masked_prefix=masked_prompt))

        return predict_prefix_probability(tokenizer, model, masked_prompt, word, masked_prefix=masked_prompt)
    ##############################
    dataset["p_target"] = dataset.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
    dataset["p_incorrect"] = dataset.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
    dataset["correct"] = dataset.apply(lambda x: x["p_target"] > x["p_incorrect"], axis=1)

    # alright, now break it to dataset_one, dataset_two
    dataset_one, dataset_two = dataset[dataset["test_type"] == 1], dataset[dataset["test_type"] == 2]
    accuracy_one = len(dataset_one[dataset_one["correct"] == True]) / len(dataset_one)
    accuracy_two = len(dataset_two[dataset_two["correct"] == True]) / len(dataset_two)
    return accuracy_one, accuracy_two


def main():
    args = parse_args()
    model_name, adversarial_filtering_mode = args.m, args.afmode
    overwrite = args.overwrite == "True"
    print("model_name, adversarial_filtering_mode: ", model_name, adversarial_filtering_mode)
    # assert that adversarial filtering mode is in ['none', 'model_specific', 'all_models']
    model, tokenizer = get_pretrained_path_template_and_model(model_name)
    
    # checkpoint_to_performance = {} # map from checkpoint -> performance on the first and second tests
    """
    Data is stored in the form:
    {
        "test_one": {
            "none": ...,
            "model_specific": ...,
            "all_models": ...
        },
        "test_two": {
            "none": ...,
            "model_specific": ...,
            "all_models": ...
        }
    """
    # determining the place to write the json results to
    new_mod_name = model_name.replace("/", "-")
    writepath = f"./results/{new_mod_name}-pretrained.json"
    print("Going to write to path: ", writepath)
    
    none_dataset, model_specific_dataset, all_models_dataset = get_three_tiers_dataset(model_name)
    print("Finished loading all datasets... Now onto evaluating accuracy metrics")
    if os.path.exists(writepath) and not overwrite:
        with open(writepath, "r") as openfile:
            prev_res = json.load(openfile)
    else:
        prev_res = {
            "test_one": {
                "none": -1,
                "model_specific": -1,
                "all_models": -1
            },
            "test_two": {
                "none": -1,
                "model_specific": -1,
                "all_models": -1
            }
        }
    
    result_dict = prev_res.copy()
    # accuracy_one_none, accuracy_two_none = get_accuracies_one_two(model, tokenizer, model_name, none_dataset)
    # accuracy_one_modspec, accuracy_two_modspec = get_accuracies_one_two(model, tokenizer, model_name, model_specific_dataset)
    # accuracy_one_all, accuracy_two_all = get_accuracies_one_two(model, tokenizer, model_name, all_models_dataset)

    for dataset, name in [(none_dataset, "none"), (model_specific_dataset, "model_specific"), (all_models_dataset, "all_models")]:
        if result_dict["test_one"][name] != -1 and result_dict["test_two"][name] != -1:
            continue
        one_res, two_res = get_accuracies_one_two(model, tokenizer, model_name, dataset)
        result_dict["test_one"][name] = one_res
        result_dict["test_two"][name] = two_res
        with open(writepath, "w") as openfile:
            json.dump(result_dict, openfile)
    

    # result_dict = {
    #     "test_one": {
    #         "none": accuracy_one_none,
    #         "model_specific": accuracy_one_modspec,
    #         "all_models": accuracy_one_all
    #     },
    #     "test_two": {
    #         "none": accuracy_two_none,
    #         "model_specific": accuracy_two_modspec,
    #         "all_models": accuracy_two_all
    #     }
    # }
    
    
    # with open(writepath, "w") as openfile:
    #     json.dump(result_dict, openfile)
    
    


if __name__ == "__main__":
    main()
    