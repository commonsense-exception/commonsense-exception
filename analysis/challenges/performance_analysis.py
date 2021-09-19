"""
The point of this file is to (1) see what models predict given a mask,
(2) calculate the precision-recall and average 
"""

import pandas as pd
import os, argparse
from tqdm import tqdm
import importlib.util

############################# Helpers! #############################

def parse_args():
    parser = argparse.ArgumentParser(description='arguments for analyzing how models perform given challenges')
    parser.add_argument('-d', help="Path to the challenges", default="../../data/winoventi_bert_large_final.tsv")
    parser.add_argument('-output_probs_path', help="Path to the outcome predicted probabilities", default="./assets/output_predicted_probabilities.tsv")
    parser.add_argument('-write_output_probs_path', help="Value to check if we're writing to the output_probs_path or not, ['True', 'False']", default="False")
    parser.add_argument('-output_aggregated_path', help="Path to the results (aggregated per model)", default="./assets/output_aggregated.tsv")
    parser.add_argument('-write_output_aggregated_path', help="Value to check if we're writing to the output_aggregated_path or not, ['True', 'False']", default="False")
    parser.add_argument('-associative_bias_registry_path', help="Path to the associative bias registry", default="./../../data/assets/associativebias_registry.tsv")

    # Brown University argument
    parser.add_argument('-dept', help="whether we're on the department machine or not", default="True")
    return parser.parse_args()

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# import the necessary functions from `maskedlm.py`
MASKEDLM_MODULE_PATH = "./../../code/maskedlm/maskedlm.py"

# function `predict_prefix_probability`
predict_prefix_probability = module_from_file('predict_prefix_probability', MASKEDLM_MODULE_PATH).predict_prefix_probability

# list of model names of interest
MODEL_NAMES = module_from_file('MODEL_NAMES', MASKEDLM_MODULE_PATH).MODEL_NAMES

# and the function to load the tokenizer and model
load_model_and_tokenizer = module_from_file('load_model_and_tokenizer', MASKEDLM_MODULE_PATH).load_model_and_tokenizer

huggingface_to_model_name = {
    "bert-base-cased": "BERT_base",
    "bert-large-cased-whole-word-masking": "BERT_large",
    "roberta-base": "RoBERTa_small",
    "roberta-large": "RoBERTa_large",
    "distilroberta-base": "DistilRoBERTa",
    "squeezebert/squeezebert-uncased": "SqueezeBERT",
    "google/mobilebert-uncased": "MobileBERT",
    "albert-base-v2": "ALBERT_base",
    "albert-large-v2": "ALBERT_large",
    "albert-xlarge-v2": "ALBERT_xlarge",
    "albert-xxlarge-v2": "ALBERT_xxlarge",
    "distilbert-base-cased": "DistilBERT"
}

model_name_to_huggingface = {
    "BERT_base": "bert-base-cased",
    "BERT_large": "bert-large-cased-whole-word-masking",
    "RoBERTa_small": "roberta-base",
    "RoBERTa_large": "roberta-large",
    "DistilRoBERTa": "distilroberta-base",
    "DistilBERT": "distilbert-base-cased",
    "SqueezeBERT": "squeezebert/squeezebert-uncased",
    "MobileBERT": "google/mobilebert-uncased",
    "ALBERT_base": "albert-base-v2",
    "ALBERT_large": "albert-large-v2",
    "ALBERT_xlarge": "albert-xlarge-v2",
    "ALBERT_xxlarge": "albert-xxlarge-v2"
}

###########################################################################

############## Functions to use models to add probabilities ###############

def add_probabilities_maskedlm(df, tokenizer, model, model_name):
    """
    this function is to add two columns, p_target_(MODEL_NAME)
    (e.g. p_target_BERT_large) and p_incorrect_(MODEL_NAME) (e.g. p_incorrect_BERT_large)
    """
    def get_probability(masked_prompt, word):
        return predict_prefix_probability(tokenizer, model, masked_prompt, word, masked_prefix=masked_prompt)
    # This is so that we don't mutate
    new_df = df.copy()
    # If these two columns are already calculated, we'll just return new_df
    p_target_col_name, p_incorrect_col_name = f"p_target_{model_name}", f"p_incorrect_{model_name}"
    existing_columns = new_df.columns
    # if the probabilities for target and other choice have been calculated, we'll just move on
    if p_target_col_name in existing_columns and p_incorrect_col_name in existing_columns:
        return new_df
    # If not, we will go ahead and calculate things
    new_df[p_target_col_name] = new_df.apply(lambda x: get_probability(x["masked_prompt"], x["target"]), axis=1)
    new_df[p_incorrect_col_name] = new_df.apply(lambda x: get_probability(x["masked_prompt"], x["incorrect"]), axis=1)
    # and then return the new thang
    return new_df


###########################################################################

############# Functions to analyze precision-recall-accuracy ##############

    
def get_mutually_passed_associativebias(to_filter_df, m_names: list):
    """
    get the samples that pass all associative bias requirement for *all
    models* in m_names (Word, Associative Bias, Alternative), and then filter
    to_filter_df to get the correct ones

    Example usage:
    get_mutually_passed_associative_bias(
        df_that_contains_examples_to_filter,
        ["BERT_base", "BERT_large", "RoBERTa_small", "RoBERTa_large"]
    )
    """
    qualified_asscbias_rows = {}
    def filter_asscbias_helper(row):
        # helper to filter out rows that don't satisfy m_names associative bias reqs
        for m in m_names:
            if not row[f"{model_name_to_huggingface[m]}"]: return False
        return True
    
    def populate_qualified_asscbias_rows(row):
        # called by qualified_unflattened_df, to populate qualified_assscbias_rows
        qualified_asscbias_rows[(row["Word"], row["Associative Bias"], row["Alternative"])] = 1

    def filter_to_filter_df_helper(row):
        return (row["Word"], row["Associative Bias"], row["Alternative"]) in qualified_asscbias_rows


    asscbiaspass_df = pd.read_csv(ASSSOCIATIVE_BIAS_REGISTRY_PATH, sep="\t")
    # get the qualified unflattened df
    qualified_unflattened_df = asscbiaspass_df[asscbiaspass_df.apply(lambda x: filter_asscbias_helper(x), axis=1)]
    # and then populate qualified_asscbias_rows
    qualified_unflattened_df.apply(lambda x: populate_qualified_asscbias_rows(x), axis=1)
    # and then, filter the to filter df based on qualified_asscbias_rows
    filtered_to_filter = to_filter_df[to_filter_df.apply(lambda x: filter_to_filter_df_helper(x), axis=1)]
    # and then return
    return filtered_to_filter
    

def get_accuracy(df, model_name, test_type):
    """
    return accuracy given dataframe and model name
    """
    p_correct_col, p_incorrect_col = f"p_target_{model_name}", f"p_incorrect_{model_name}"
    new_df = df.copy()
    new_df = new_df[new_df["test_type"] == test_type]
    if len(new_df) == 0:
        print("Error caught: Model == {}, test_type == {}".format(model_name, test_type))
        return 0
    df_correct = new_df[new_df.apply(lambda x: x[p_correct_col] > x[p_incorrect_col], axis=1)]
    return len(df_correct) / len(new_df)


def generate_binary_cls_aggregate(df):
    """
    generate a table of this format:
                all_samples_1   all_samples_2   all_samples_3   all_samples_4     pass_asscbias_w_BERT_large...      all_pass_asscbias...
    BERT_base
    BERT_large
    """
    list_of_rows = []
    # get the df of samples that pass all associative biases
    pass_all = get_mutually_passed_associativebias(df, MODEL_NAMES)
    pass_bert_large = get_mutually_passed_associativebias(df, ["BERT_large"])
    # types: (1) "stereotypical challenges", (2) adversarial challenges (see paper)
    all_types = [1, 2]
    for m in MODEL_NAMES:
        new_row = [m]
        for t_type in all_types:
            # get all the rows of examples in the challenge set,
            # and get the number of passes
            new_row.append(get_accuracy(pass_bert_large, m, t_type)) # all
            all_specific_type = pass_bert_large[pass_bert_large["test_type"] == t_type]
            new_row.append(len(all_specific_type))

            # get all the rows of examples that pass the associative bias of THAT specific model
            # and get the number of passes
            pass_itself = get_mutually_passed_associativebias(df, [m])
            new_row.append(get_accuracy(pass_itself, m, t_type)) # pass_itself
            pass_itself_specific_type = pass_itself[pass_itself["test_type"] == t_type]
            new_row.append(len(pass_itself_specific_type)) # num_itself

            # get all the rows of examples that pass ALL models, and get the number of passes
            new_row.append(get_accuracy(pass_all, m, t_type))
            intersection_specific_type = pass_all[pass_all["test_type"] == t_type]
            new_row.append(len(intersection_specific_type))
        
        # and then append this new row to list of rows
        list_of_rows.append(new_row)
        new_row = []
        
    # and then make a new dataframe
    column_names = ["model_name"]
    for t_type in all_types:
        for stuff in ["all", "num_all", "pass_itself", "num_itself", "pass_intersection", "num_intersection"]:
            column_names.append(stuff + "_{}".format(t_type))


    new_df = pd.DataFrame(list_of_rows, columns=column_names)
    return new_df
        






###########################################################################

if __name__ == "__main__":
    args = parse_args()
    ASSSOCIATIVE_BIAS_REGISTRY_PATH = args.associative_bias_registry_path

    # If we have calculated the p for at least one model before
    if os.path.exists(args.output_probs_path):
        # Then load from previous
        challenge_df = pd.read_csv(args.output_probs_path, sep="\t")
    else:
        # If not, we'll take it directly from the challenge dataset
        challenge_df = pd.read_csv(args.d, sep="\t")

    added_probabilities = challenge_df.copy()
    # And then, for each model, we'll add for each row the probabilities that that model
    # associates with target and incorrect choices
    for m_name in tqdm(MODEL_NAMES):
        tknzr, mdl = load_model_and_tokenizer(m_name)
        # implement a count system so we preserve challenge_df for whatever need
        added_probabilities = add_probabilities_maskedlm(added_probabilities, tknzr, mdl, m_name)
        # And then, output it to the path
        if args.write_output_probs_path == "True": # wacky engineering sorry lol
            added_probabilities.to_csv(args.output_probs_path, sep="\t", index=False)

    ####### okay, now is the big step to calculate the aggregated stats #######

    ## Okay, now we're calculating the aggregate df for binary
    binary_aggregate_df = generate_binary_cls_aggregate(added_probabilities)

    # and then outputting that
    if args.write_output_aggregated_path == "True": # wacky engineering sorry lol
        binary_aggregate_df.to_csv(args.output_aggregated_path, sep="\t", index=False)