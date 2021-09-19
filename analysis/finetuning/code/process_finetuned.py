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

# chaos engineering
TRAIN, TESTS, ASSOCIATIVE_BIAS_REGISTRY = None, None, None

MODELS = [
    "bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base",\
    "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",\
    "allenai/longformer-large-4096", "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2"
]

def get_accuracy_evaluation_data(finetune_type):
    # return: train, test, associative_bias_registry
    train_path = os.getcwd() + f"/../../data/finetune/{finetune_type}/train.txt"
    # now for tests, we have two scenarios:
    if finetune_type == "winoventi/both_generic_exception":
        # if its both, then its straight forward
        test_paths = [os.getcwd() + f"/../../data/finetune/{finetune_type}/test.txt"]
    else:
        # if not, then it means that we're going to evaluate on three different things:
        # -- (1) the held out test exception schemas
        # -- (2) all the generic schemas
        # -- (3) both generic and exception schemas (50-50)
        test_paths = [
            os.getcwd() + f"/../../data/finetune/{finetune_type}/unseen-exception-schemas-test.txt",
            os.getcwd() + f"/../../data/finetune/{finetune_type}/only-generics-test.txt",
            os.getcwd() + f"/../../data/finetune/{finetune_type}/both-generic-exception-test.txt"
        ]

    
    # test_path = os.getcwd() + f"/../../data/finetune/{finetune_type}/test.txt"
    # test_path = os.getcwd() + "/../../data/finetune/winoventi/both_generic_exception/test.txt"
    adversarial_registry_path = os.getcwd() + "/../../data/process/associativebias_registry.tsv"
    assert os.path.exists(train_path)
    for test_path in test_paths:
        assert os.path.exists(test_path), f"path does not exist; {test_path}"
    assert os.path.exists(adversarial_registry_path)
    return pd.read_csv(train_path, sep="\t", names=["text"])["text"].values.tolist(),\
            [pd.read_csv(test_path, sep="\t", names=["text"])["text"].values.tolist() for test_path in test_paths],\
                pd.read_csv(adversarial_registry_path, sep="\t")


###### HELPERS ######

def parse_args():
    parser = argparse.ArgumentParser(description='finetuning BERT')
    parser.add_argument('-m', help="model name. {}".format(MODELS), default='bert-base-cased')
    parser.add_argument('-afmode', help="adversarial filtering or not ['none', 'model_specific', 'all_models']",\
                                                    default='model_specific')
    parser.add_argument('-evalmode', help="evaluation mode: ['qualitative', 'quantitative']", default='quantitative')
    parser.add_argument('-finetunemode', help="finetune mode: ['winoventi/both_generic_exception', 'winoventi/only_exception']",\
                                         default='winoventi/only_exception')
    # finetuning args
    parser.add_argument('-dept', help="whether we're on the department machine or not", default="True")
    # top or all - if top, then we're only evaluating the most recent checkpoint
    parser.add_argument('-top', help='evaluate the max/latest checkpoint, or just the most recent one', default="top")
    parser.add_argument('-overwrite', help="overwrite previous processed results or not", default="not")
    return parser.parse_args()

#########################


def get_pretrained_path_template_and_model(model_name, adversarial_filtering_mode, finetune_mode="winoventi/both_generic_exception"):
    assert model_name in MODELS
    assert adversarial_filtering_mode in ['none', 'model_specific', 'all_models']
    path = f"./output/{finetune_mode}/{model_name}"
    if not os.path.exists(path):
        print("Path unfortunately does not exist")
        checkpoints = []
    else:
        checkpoints = [int(e.replace("checkpoint-", "")) for e in os.listdir(path) if "checkpoint" in e]
    # model_path = path + "/checkpoint-{}".format(max(content))
    path_template = path + "/checkpoint-{}"
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
        "albert-xxlarge-v2": (AlbertTokenizer, AlbertForMaskedLM)
    }
    tokenizer_module, model_module = name_to_tokenizer_model[model_name]
    return path_template, checkpoints, model_module, tokenizer_module

# Done on dept: albert-base-v2 (d), google/mobilebert-uncased, roberta-base (d), squeezebert/squeezebert-uncased (d),
#               distilroberta-base (d)
# Hung: roberta-large, allenai/longformer-base-4096, allenai/longformer-large-4096

def evaluate_winoventi_qualitative_performance(model, tokenizer, model_name, adversarial_filtering_mode="model_specific", checkpoint_name=None):
    ############### defining helper function and helper variables ###############
    def get_prediction(masked_prompt):
        return one_mask_generation(tokenizer, model, masked_prompt, masked_sentence=masked_prompt)

    def is_train(row):
        mask_replaced = row["masked_prompt"].replace("[MASK]", row["target"])
        return mask_replaced in TRAIN

    def is_test(row):
        mask_replaced = row["masked_prompt"].replace("[MASK]", row["target"])
        return mask_replaced in TEST

    def populate_set(row, to_populate_set):
        to_populate_set.add((row["Word"], row["Associative Bias"], row["Alternative"]))

    def in_set(row, populated_set):
        return (row["Word"], row["Associative Bias"], row["Alternative"]) in populated_set

    def pass_all_associative_bias_tests(row):
        return_bool = True
        for m in MODELS:
            return_bool = return_bool and row[m]
        return return_bool

    #############################################################################
    print("Received call to qualitatively evaluate {}_{}".format(model_name, checkpoint_name))
    # first, get the challenge
    challenge_path = os.getcwd() + "/../../data/final/winoventi_bert_large_final.tsv"
    challenge = pd.read_csv(challenge_path, sep="\t")

    challenge_size_before = challenge.shape[0]


    # alright. now make the challenge to be adversarial depending on whether it is model specific or all_models
    if adversarial_filtering_mode == "model_specific":
        # filter challenge based on it being in the adversarial ones or not for this model
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY = ASSOCIATIVE_BIAS_REGISTRY[ASSOCIATIVE_BIAS_REGISTRY[model_name]]
        passed = set()
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: populate_set(x, passed), axis=1)
        challenge = challenge[challenge.apply(lambda x: in_set(x, passed), axis=1)]
        

        
    if adversarial_filtering_mode == "all_models":
        # filter challenge based on it being in the adversarial ones or not across all models
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY = ASSOCIATIVE_BIAS_REGISTRY[ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: pass_all_associative_bias_tests(x), \
                                                                                                                        axis=1)]
        passed = set()
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: populate_set(x, passed), axis=1)
        
        

    
    challenge_size_after = challenge.shape[0]
    

        

    challenge_train, challenge_test = challenge[challenge.apply(lambda x: is_train(x), axis=1)],\
                                        challenge[challenge.apply(lambda x: is_test(x, 0), axis=1)]


    print("Filtered out: ", challenge_size_before - challenge_size_after)
    # alright, if the model name has "albert" in it, then we'll do a special prediction thing on top of it and then
    # we will do a special thing
    qualitative_path = os.getcwd() + "/results/qualitative"
    if not os.path.exists(qualitative_path):
        os.mkdir(qualitative_path, mode=0o775)
    challenge_train["predicted"] = challenge_train.apply(lambda x: get_prediction(x["masked_prompt"]), axis=1)
    challenge_train.to_csv(qualitative_path + "/{}_{}_train_predictions.csv".format(model_name, checkpoint_name))
    challenge_test["predicted"] = challenge_test.apply(lambda x: get_prediction(x["masked_prompt"]), axis=1)
    challenge_test.to_csv(qualitative_path + "/{}_{}_test_predictions.csv".format(model_name, checkpoint_name))

    


def evaluate_winoventi_performance(model, tokenizer, model_name, adversarial_filtering_mode="model_specific", finetunemode="winoventi/only_exception"):
    ############### defining helper function and helper variables ###############
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

    # def get_prediction(masked_prompt):
    #     return one_mask_generation(tokenizer, model, masked_prompt, masked_sentence=masked_prompt)

    def is_train(row):
        mask_replaced = row["masked_prompt"].replace("[MASK]", row["target"])
        return mask_replaced in TRAIN

    def is_test(row, test_index):
        mask_replaced = row["masked_prompt"].replace("[MASK]", row["target"])
        return mask_replaced in TESTS[test_index]

    def populate_set(row, to_populate_set):
        to_populate_set.add((row["Word"], row["Associative Bias"], row["Alternative"]))

    def in_set(row, populated_set):
        return (row["Word"], row["Associative Bias"], row["Alternative"]) in populated_set

    def pass_all_associative_bias_tests(row):
        return_bool = True
        for m in MODELS:
            return_bool = return_bool and row[m]
        return return_bool

    #############################################################################
    # first, get the challenge
    challenge_path = os.getcwd() + "/../../data/final/winoventi_bert_large_final.tsv"
    challenge = pd.read_csv(challenge_path, sep="\t")

    challenge_size_before = challenge.shape[0]


    # alright. now make the challenge to be adversarial depending on whether it is model specific or all_models
    if adversarial_filtering_mode == "model_specific":
        # filter challenge based on it being in the adversarial ones or not for this model
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY = ASSOCIATIVE_BIAS_REGISTRY[ASSOCIATIVE_BIAS_REGISTRY[model_name]]
        passed = set()
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: populate_set(x, passed), axis=1)
        challenge = challenge[challenge.apply(lambda x: in_set(x, passed), axis=1)]
        

    if adversarial_filtering_mode == "all_models":
        # filter challenge based on it being in the adversarial ones or not across all models
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY = ASSOCIATIVE_BIAS_REGISTRY[ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: pass_all_associative_bias_tests(x), \
                                                                                                                        axis=1)]
        passed = set()
        FILTERED_ASSOCIATIVE_BIAS_REGISTRY.apply(lambda x: populate_set(x, passed), axis=1)
        challenge = challenge[challenge.apply(lambda x: in_set(x, passed), axis=1)]

    
    challenge_size_after = challenge.shape[0]
    
    # This is really chaotic engineering I'm sorry
    if finetunemode == "winoventi/both_generic_exception":
        challenge_train, challenge_test = challenge[challenge.apply(lambda x: is_train(x), axis=1)],\
                                            challenge[challenge.apply(lambda x: is_test(x, 0), axis=1)]


        print("Filtered out: ", challenge_size_before - challenge_size_after)

        # alright. now predict p_target and p_incorrect for train
        challenge_train["p_target_train"] = challenge_train.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
        challenge_train["p_incorrect_train"] = challenge_train.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
        challenge_train["correct_train"] = challenge_train.apply(lambda x: x["p_target_train"] > x["p_incorrect_train"], axis=1)
        # alright. now predict p_target and p_incorrect for test
        challenge_test["p_target_test"] = challenge_test.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
        challenge_test["p_incorrect_test"] = challenge_test.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
        challenge_test["correct_test"] = challenge_test.apply(lambda x: x["p_target_test"] > x["p_incorrect_test"], axis=1)
        # alright, now lets break it to challenge_one_train, challenge_one_test, challenge_two_train, challenge_two_test
        challenge_one_train, challenge_two_train = challenge_train[challenge_train["test_type"] == 1],\
                                                    challenge_train[challenge_train["test_type"] == 2]
        challenge_one_test, challenge_two_test = challenge_test[challenge_test["test_type"] == 1],\
                                                    challenge_test[challenge_test["test_type"] == 2],

        accuracy_one_train = len(challenge_one_train[challenge_one_train["correct_train"] == True]) / len(challenge_one_train)
        accuracy_one_test = len(challenge_one_test[challenge_one_test["correct_test"] == True]) / len(challenge_one_test)

        accuracy_two_train = len(challenge_two_train[challenge_two_train["correct_train"] == True]) / len(challenge_two_train)
        accuracy_two_test = len(challenge_two_test[challenge_two_test["correct_test"] == True]) / len(challenge_two_test)

        result_dict = {
            "accuracy_one_train": accuracy_one_train,
            "accuracy_two_train": accuracy_two_train,
            "accuracy_one_test": accuracy_one_test,
            "accuracy_two_test": accuracy_two_test
        }
        return result_dict

    if finetunemode == "winoventi/only_exception":
        challenge_train = challenge[challenge.apply(lambda x: is_train(x), axis=1)]
        # if not, then it means that we're going to evaluate on three different things:
        # -- (1) the held out test exception schemas - 435
        # -- (2) all the generic schemas - 2176
        # -- (3) both generic and exception schemas (50-50) - 435 * 2
        challenge_test_heldout_exception = challenge[challenge.apply(lambda x: is_test(x, 0), axis=1)]
        challenge_test_all_generics = challenge[challenge.apply(lambda x: is_test(x, 1), axis=1)]
        challenge_test_both_generic_exception = challenge[challenge.apply(lambda x: is_test(x, 2), axis=1)]

        # alright. now predict p_target and p_incorrect for train
        challenge_train["p_target_train"] = challenge_train.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
        challenge_train["p_incorrect_train"] = challenge_train.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
        challenge_train["correct_train"] = challenge_train.apply(lambda x: x["p_target_train"] > x["p_incorrect_train"], axis=1)
        
        accuracy_train = len(challenge_train[challenge_train["correct_train"] == True]) / len(challenge_train)

        # alright, now predict p_target and p_incorrect for test_heldout_exception
        challenge_test_heldout_exception["p_target_test_heldout_exception"] = challenge_test_heldout_exception.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
        challenge_test_heldout_exception["p_incorrect_test_heldout_exception"] = challenge_test_heldout_exception.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
        challenge_test_heldout_exception["correct_test_heldout_exception"] = challenge_test_heldout_exception.apply(lambda x: x["p_target_test_heldout_exception"] > x["p_incorrect_test_heldout_exception"], axis=1)
        # because this is only exceptions (type two), we will
        accuracy_test_heldout_exception = len(challenge_test_heldout_exception[challenge_test_heldout_exception['correct_test_heldout_exception'] == True]) / len(challenge_test_heldout_exception)

        # and the same thing for the remaining two tests
        challenge_test_all_generics["p_target_test_all_generics"] = challenge_test_all_generics.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
        challenge_test_all_generics["p_incorrect_test_all_generics"] = challenge_test_all_generics.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
        challenge_test_all_generics["correct_test_all_generics"] = challenge_test_all_generics.apply(lambda x: x["p_target_test_all_generics"] > x["p_incorrect_test_all_generics"], axis=1)
        # because this is only generics (type one), we will
        accuracy_test_all_generics = len(challenge_test_all_generics[challenge_test_all_generics['correct_test_all_generics'] == True]) / len(challenge_test_all_generics)
        

        challenge_test_both_generic_exception["p_target_test_both_generic_exception"] = challenge_test_both_generic_exception.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
        challenge_test_both_generic_exception["p_incorrect_test_both_generic_exception"] = challenge_test_both_generic_exception.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
        challenge_test_both_generic_exception["correct_test_both_generic_exception"] = challenge_test_both_generic_exception.apply(lambda x: x["p_target_test_both_generic_exception"] > x["p_incorrect_test_both_generic_exception"], axis=1)
        # alright, but for this one we have tw different types so we will have to split it out
        challenge_one_test_both_generic_exception, challenge_two_test_both_generic_exception = challenge_test_both_generic_exception[challenge_test_both_generic_exception["test_type"] == 1],\
                                                                                        challenge_test_both_generic_exception[challenge_test_both_generic_exception["test_type"] == 2]
        accuracy_one_test_both_generic_exception = len(challenge_one_test_both_generic_exception[challenge_one_test_both_generic_exception["correct_test_both_generic_exception"] == True]) / \
                                                     len(challenge_one_test_both_generic_exception)
        accuracy_two_test_both_generic_exception = len(challenge_two_test_both_generic_exception[challenge_two_test_both_generic_exception["correct_test_both_generic_exception"] == True]) / \
                                                     len(challenge_two_test_both_generic_exception)

        result_dict = {
            "accuracy_train": accuracy_train,
            "accuracy_test_heldout_exception": accuracy_test_heldout_exception,
            "accuracy_test_all_generics": accuracy_test_all_generics,
            "accuracy_one_test_both_generic_exception": accuracy_one_test_both_generic_exception,
            "accuracy_two_test_both_generic_exception": accuracy_two_test_both_generic_exception
        }

        return result_dict




    

def evaluate_checkpoint(path_template, checkpoint, model_module, tokenizer_module, model_name, adversarial_filtering_mode="model_specific", eval_mode="qualitative", finetunemode="winoventi/only_exception"):
    # this means that this is the pre-trained case
    if checkpoint == 0:
        model = model_module.from_pretrained(model_name).eval()
        tokenizer = tokenizer_module.from_pretrained(model_name)
    # this is the fine-tuned case
    else:
        path = path_template.format(checkpoint)
        model = model_module.from_pretrained(path).eval()
        tokenizer = tokenizer_module.from_pretrained(path)
    if eval_mode == "quantitative":
        return evaluate_winoventi_performance(model, tokenizer, model_name, adversarial_filtering_mode, finetunemode)
    else:
        return evaluate_winoventi_qualitative_performance(model, tokenizer, model_name, adversarial_filtering_mode, checkpoint)


def main():
    global TRAIN, TESTS, ASSOCIATIVE_BIAS_REGISTRY
    args = parse_args()
    model_name, adversarial_filtering_mode, eval_mode = args.m, args.afmode, args.evalmode
    finetune_mode = args.finetunemode
    top = args.top == "top"
    overwrite = args.overwrite == "True"
    TRAIN, TESTS, ASSOCIATIVE_BIAS_REGISTRY = get_accuracy_evaluation_data(args.finetunemode)
    print("model_name, adversarial_filtering_mode: ", model_name, adversarial_filtering_mode)
    # assert that adversarial filtering mode is in ['none', 'model_specific', 'all_models']
    path_template, checkpoints, model_module, tokenizer_module = get_pretrained_path_template_and_model(model_name, adversarial_filtering_mode, finetune_mode)
    # checkpoint_to_performance = {} # map from checkpoint -> performance on the first and second tests
    if eval_mode == "quantitative":
        if not top:
            """
            Data is stored in the form:
            {
                1000: {
                    "accuracy_one_train": 0.9,
                    "accuracy_two_train": 
                }
            }
            """
            new_rows = []
            print("We got here yeehaw poopoogang, len(checkpoints): ", len(checkpoints))
            # determining the place to write the json results to
            new_mod_name = model_name.replace("/", "-")
            writepath = f"./results/{args.finetunemode}/{new_mod_name}-{adversarial_filtering_mode}.json"
            print("Writing results to path: ", writepath)
            if not os.path.exists(writepath):
                with open(writepath, "w") as openfile:
                    json.dump({}, openfile)
            with open(writepath, "r") as openfile:
                prev_res_dict = json.load(openfile)
            
            sorted_checkpoints = [0] + list(sorted(checkpoints))
            for checkpoint in sorted_checkpoints:
                # this is so that we'll only process every other checkpoint - to speed things up a lil
                done = (checkpoint in prev_res_dict or str(checkpoint) in prev_res_dict)
                if not overwrite and done:
                    print(f"checkpoint {checkpoint} previously processed. continuing...")
                    continue
                
                
                print("processing checkpoint : ", checkpoint)
                return_dict = evaluate_checkpoint(path_template, checkpoint, model_module, tokenizer_module, \
                                        model_name, adversarial_filtering_mode, eval_mode, finetune_mode)
                prev_res_dict[checkpoint] = return_dict
                # new_rows.append([checkpoint, accuracy_one_train, accuracy_two_train, accuracy_one_test, accuracy_two_test])
                # print("[checkpoint, accuracy_one_train, accuracy_two_train, accuracy_one_test, accuracy_two_test]", \
                #         [checkpoint, accuracy_one_train, accuracy_two_train, accuracy_one_test, accuracy_two_test])
                # to_return_df = pd.DataFrame(new_rows, columns=["checkpoint", "accuracy_one_train", "accuracy_two_train", \
                #                                                             "accuracy_one_test", "accuracy_two_test"])
                # to_return_df.to_csv(f"./results/{args.finetunemode}/{new_mod_name}-{adversarial_filtering_mode}.tsv", sep="\t", index=False)
                # counter += 1
                # print(to_return_df)
                with open(writepath, "w") as openfile:
                    json.dump(prev_res_dict, openfile)
        else:
            """
            Data is stored in the form:
            {
                "bert-base-cased": {
                    "checkpoint": 35000
                    "accuracy_one_train": 100
                    "accuracy_two_train": 100
                    "accuracy_one_test": 90
                    "accuracy_two_test": 10
                },
                "bert-large-...": {
                    ...
                }
            }
            path to save at: f"./results/{args.finetunemode}/{adversarial_filtering_mode}-max.json"
            """
            res_path = f"./results/{args.finetunemode}/{adversarial_filtering_mode}-max.json"
            if not os.path.exists(res_path):
                with open(res_path, "w") as openfile:
                    json.dump({}, openfile)
            checkpoint = sorted(checkpoints)[-1]
            print("Processing the top checkpoint: ", checkpoint)
            return_dict = evaluate_checkpoint(path_template, checkpoint, model_module, tokenizer_module, \
                                        model_name, adversarial_filtering_mode, eval_mode, finetune_mode)
            return_dict["checkpoint"] = checkpoint

            with open(res_path, "r") as openfile:
                aggregate = json.load(openfile)

            aggregate[model_name] = return_dict
            with open(res_path, "w") as openfile:
                json.dump(aggregate, openfile)
    else:
        sorted_checkpoints = [0] + list(sorted(checkpoints))
        for checkpoint in sorted_checkpoints:
            evaluate_checkpoint(path_template, checkpoint, model_module, tokenizer_module, model_name, adversarial_filtering_mode, eval_mode, finetune_mode)
    
    


if __name__ == "__main__":
    main()
    