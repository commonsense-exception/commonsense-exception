#############################################  IMPORT STUFF  #############################################
import pandas as pd
import numpy as np
import spacy
import importlib.util
import sys
import math
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# Load pre-trained model (weights)
gpt_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
gpt_model.eval()
gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


# helper function to help load things from BERT folder
def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# load function from maskedlm to process things
MASKEDLM_MODULE_PATH = "../../../../../code/maskedlm/maskedlm.py"


maskedlm_generation = module_from_file('predict_prefix_probability', MASKEDLM_MODULE_PATH)
predict_prefix_probability = maskedlm_generation.predict_prefix_probability

one_mask_module = module_from_file('one_mask_generation', MASKEDLM_MODULE_PATH)
one_mask_generation = one_mask_module.one_mask_generation

# load dict of different models for maskedlm
dict_module = module_from_file('TOKENIZER_MODEL_DICT', MASKEDLM_MODULE_PATH)
TOKENIZER_MODEL_DICT = dict_module.TOKENIZER_MODEL_DICT


##########################################  END OF IMPORT STUFF  ##########################################

#############################################  HELPER STUFF  ##############################################

def gpt_score(sentence):
    tokenize_input = gpt_tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([gpt_tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=gpt_model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


##########################################  END OF HELPER STUFF  ##########################################


# step 1: predict the probability of the generic context and the exception context
FILES = ["batch0_0.csv"]
PROBABILITY_PATHS = ["batch0_0_with_probs.csv"]
DFS = [pd.read_csv(f, sep=",") for f in FILES]

def load_from_file(df, i):
    return pd.read_csv(PROBABILITY_PATHS[i], sep="\t")
    

def add_probabilities(tokenizer, model, model_name, df, i, loadFromFile=True):
    """
    Function to add the probability of the *special word* in the masked position in the context.
    TODO: Find the correlation between this and probabilities in experiment 3 and 4
    """
    new_df = df.copy()
    print("COLUMNS: ", new_df.columns)
    if loadFromFile:
        return new_df
    def find_probability(masked_nl_context_or_outcome, word):
        if pd.isnull(word) or word == "" or word == "N/A": return 0
        return predict_prefix_probability(tokenizer, model, masked_nl_context_or_outcome, word, masked_prefix=masked_nl_context_or_outcome)
    cols = ["generic_context", "exception_context", "generic_outcome", "exception_outcome"]
    for col in cols:
        if "context" in col: right_masked_col_name = "nl_context_masked"
        else: right_masked_col_name = "nl_outcome_masked"
        if "p_" + col + "_(MODEL_NAME)".replace("(MODEL_NAME)", model_name) not in new_df.columns:
            new_df["p_" + col + "_(MODEL_NAME)".replace("(MODEL_NAME)", model_name)] = new_df.apply(lambda x: find_probability(x[right_masked_col_name], x[col]), axis=1)
    print("model_name: ", model_name)
    print("new_df: ", new_df)
    return new_df


def record_dfs(list_dfs, list_paths):
    """
    Function to record DFS
    """
    assert len(list_dfs) == len(list_paths)
    for i, df in enumerate(list_dfs):
        df.to_csv(list_paths[i], index=False, sep="\t")


def add_pos(df, i, loadFromFile=True):
    """
    Function to determine the part of speech of the special word in the context collected from annotators
    """
    new_df = df.copy()
    if loadFromFile:
        return new_df
    # load spacy stuff
    nlp = spacy.load("en_core_web_lg")
    
    # first, get the index of the mask
    def get_pos(row, col_name):
        masked_sentence, word = row["nl_context_masked"], row[col_name]
        replaced = masked_sentence.replace("[MASK]", word)
        processed = nlp(replaced)
        for ele in processed:
            if ele.text == word: return ele.pos_
    cols = ["generic_context", "exception_context"]
    for col in cols:
        new_df["pos_" + col] = new_df.apply(lambda x: get_pos(x, col), axis=1)
    print("new_df: ", new_df)
    return new_df
    

def add_nl_outcome(df, i, loadFromFile=True):
    """
    Function to add the mask to the outcome (the second sentence) so that we can start doing shit
    """
    new_df = df.copy()
    if loadFromFile:
        return new_df
    singular, plural = "The (WORD) is (GENERIC).", "The (WORD) are (GENERIC)."
    def choose_better_prefix(row):
        word, generic = row["Word"], row["generic_outcome"]
        s_sing = singular.replace("(WORD)", word).replace("(GENERIC)", generic)
        s_plural = plural.replace("(WORD)", word).replace("(GENERIC)", generic)
        if gpt_score(s_sing) < gpt_score(s_plural):
            to_return = singular.replace("(WORD)", word).replace("(GENERIC)", "[MASK]")
        else:
            to_return = plural.replace("(WORD)", word).replace("(GENERIC)", "[MASK]")
            print("output: ", to_return)
        return to_return
    new_df["nl_outcome_masked"] = new_df.apply(lambda x: choose_better_prefix(x), axis=1)
    print("new_df: ", new_df)
    return new_df
    

def add_pass_experiment_maskedlm(tokenizer, model, model_name, df, i, loadFromFile=False):
    def get_probability(masked_prompt, word):
        if pd.isnull(word) or word == "" or word == "N/A": return 0
        return predict_prefix_probability(tokenizer, model, masked_prompt, word, masked_prefix=masked_prompt)
    
    new_df = df.copy()
    if loadFromFile:
        return new_df

    print("Model name: ", model_name)
    
    
    def pass_experiment_one(row):
        """
        experiment one: GENERIC CONTEXT -> GENERIC OUTCOME
        - Jacob thinks the apple is [<delicious>/rotten]. The apple is [MASK].
        (a) [MASK] == edible
        (b) [MASK] == inedible
        """
        context = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        prompt = context + " " + row["nl_outcome_masked"]
        # print("begin ---")
        p_generic_outcome_given_context = get_probability(prompt, row["generic_outcome"])
        p_exception_outcome_given_context = get_probability(prompt, row["exception_outcome"])
        # print("result: ---", p_generic_outcome_given_context > p_exception_outcome_given_context)
        # print("end ---")
        if p_generic_outcome_given_context > p_exception_outcome_given_context: return True
        return False

    def pass_experiment_one_prime(row):
        """
        experiment one: GENERIC CONTEXT -> GENERIC OUTCOME
        - Jacob thinks the apple is [<delicious>/rotten]. The apple is [MASK].
        (a) [MASK] == edible
        (b) [MASK] == inedible
        """
        context = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        prompt = row["nl_outcome_masked"] + " " + context
        # print("begin ---")
        p_generic_outcome_given_context = get_probability(prompt, row["generic_outcome"])
        p_exception_outcome_given_context = get_probability(prompt, row["exception_outcome"])
        # print("result: ---", p_generic_outcome_given_context > p_exception_outcome_given_context)
        # print("end ---")
        if p_generic_outcome_given_context > p_exception_outcome_given_context: return True
        return False

    def pass_experiment_two(row):
        """
        experiment two: EXCEPTION CONTEXT -> EXCEPTION OUTCOME
        - Jacob thinks the apple is [delicious/<rotten>]. The apple is [MASK].
        (a) [MASK] == edible
        (b) [MASK] == inedible
        """
        context = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        prompt = context + " " + row["nl_outcome_masked"]
        # print("begin ---")
        p_generic_outcome_given_context = get_probability(prompt, row["generic_outcome"])
        p_exception_outcome_given_context = get_probability(prompt, row["exception_outcome"])
        # print("result: ---", p_generic_outcome_given_context > p_exception_outcome_given_context)
        # print("end ---")
        if p_exception_outcome_given_context > p_generic_outcome_given_context: return True
        return False

    def pass_experiment_two_prime(row):
        """
        experiment two: EXCEPTION CONTEXT -> EXCEPTION OUTCOME
        - Jacob thinks the apple is [delicious/<rotten>]. The apple is [MASK].
        (a) [MASK] == edible
        (b) [MASK] == inedible
        """
        context = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        prompt = row["nl_outcome_masked"] + " " + context
        # print("begin ---")
        p_generic_outcome_given_context = get_probability(prompt, row["generic_outcome"])
        p_exception_outcome_given_context = get_probability(prompt, row["exception_outcome"])
        # print("result: ---", p_generic_outcome_given_context > p_exception_outcome_given_context)
        # print("end ---")
        if p_exception_outcome_given_context > p_generic_outcome_given_context: return True
        return False
    
    def pass_experiment_three(row):
        """
        experiment three: GENERIC OUTCOME -> GENERIC CONTEXT
        - Jacob thinks the apple is [MASK]. The apple is [<edible>/inedible].
        (a) [MASK] == delicious
        (b) [MASK] == rotten
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        prompt = context + " " + outcome
        p_generic_context_given_outcome = get_probability(prompt, row["generic_context"])
        p_exception_context_given_outcome = get_probability(prompt, row["exception_context"])
        if p_generic_context_given_outcome > p_exception_context_given_outcome: return True
        return False

    def pass_experiment_four(row):
        """
        experiment four: EXCEPTION OUTCOME -> EXCEPTION CONTEXT
        - Jacob thinks the apple is [MASK]. The apple is [edible/<inedible>]
        (a) [MASK] == delicious
        (b) [MASK] == rotten
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        prompt = context + " " + outcome
        p_generic_context_given_outcome = get_probability(prompt, row["generic_context"])
        p_exception_context_given_outcome = get_probability(prompt, row["exception_context"])
        if p_exception_context_given_outcome > p_generic_context_given_outcome: return True
        return False

    def pass_experiment_three_prime(row):
        """
        experiment three: GENERIC OUTCOME -> GENERIC CONTEXT
        - The apple is [<edible>/inedible]. Jacob thinks the apple is [MASK].
        (a) [MASK] == delicious
        (b) [MASK] == rotten
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        prompt = outcome + " " + context
        p_generic_context_given_outcome = get_probability(prompt, row["generic_context"])
        p_exception_context_given_outcome = get_probability(prompt, row["exception_context"])
        if p_generic_context_given_outcome > p_exception_context_given_outcome: return True
        return False

    def pass_experiment_four_prime(row):
        """
        experiment four: EXCEPTION OUTCOME -> EXCEPTION CONTEXT
        - The apple is [edible/<inedible>]. Jacob thinks the apple is [MASK].
        (a) [MASK] == delicious
        (b) [MASK] == rotten
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        prompt = context + " " + outcome
        p_generic_context_given_outcome = get_probability(prompt, row["generic_context"])
        p_exception_context_given_outcome = get_probability(prompt, row["exception_context"])
        if p_exception_context_given_outcome > p_generic_context_given_outcome: return True
        return False
    
    if model_name + "_pass_experiment_one_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_one_maskedlm"] = new_df.apply(lambda x: pass_experiment_one(x), axis=1)
    if model_name + "_pass_experiment_one_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_one_prime_maskedlm"] = new_df.apply(lambda x: pass_experiment_one_prime(x), axis=1)
    if model_name + "_pass_experiment_two_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_two_maskedlm"] = new_df.apply(lambda x: pass_experiment_two(x), axis=1)
    if model_name + "_pass_experiment_two_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_two_prime_maskedlm"] = new_df.apply(lambda x: pass_experiment_two_prime(x), axis=1)
    if model_name + "_pass_experiment_three_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_three_maskedlm"] = new_df.apply(lambda x: pass_experiment_three(x), axis=1)
    if model_name + "_pass_experiment_four_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_four_maskedlm"] = new_df.apply(lambda x: pass_experiment_four(x), axis=1)
    if model_name + "_pass_experiment_three_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_three_prime_maskedlm"] = new_df.apply(lambda x: pass_experiment_three_prime(x), axis=1)
    if model_name + "_pass_experiment_four_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_pass_experiment_four_prime_maskedlm"] = new_df.apply(lambda x: pass_experiment_four_prime(x), axis=1)
    print("new_df: ", new_df)
    return new_df


def add_pass_experiment_multiplechoice(tokenizer, model, model_name, df, i, loadFromFile=True):
    new_df = df.copy()
    if loadFromFile:
        return new_df
    

    def pass_experiment_one(row):
        """
        experiment one: GENERIC CONTEXT -> GENERIC OUTCOME.
        - Jacob thinks the apple is [ <delicious> / rotten].
        (a) The apple is edible. <
        (b) The apple is inedible
        """
        prompt = row["nl_context_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["generic_context"])
        correct_choice = row["nl_outcome_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["generic_outcome"])
        incorrect_choice = row["nl_outcome_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["exception_outcome"])
        passed, _ = correctly_classify(tokenizer, model, prompt, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_two(row):
        """
        experiment two: EXCEPTION CONTEXT -> EXCEPTION OUTCOME.
        - Jacob thinks the apple is [delicious/ <rotten> ]
        (a) The apple is edible.
        (b) The apple is inedible <
        """
        prompt = row["nl_context_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["exception_context"])
        correct_choice = row["nl_outcome_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["exception_outcome"])
        incorrect_choice = row["nl_outcome_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["generic_outcome"])
        passed, _ = correctly_classify(tokenizer, model, prompt, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_three_prime(row):
        """
        experiment three: GENERIC OUTCOME -> GENERIC CONTEXT.
        The apple is [<edible>/inedible].
        (a) Jacob thinks the apple is delicious. <
        (b) Jacob thinks the apple is rotten.
        """
        prompt = row["nl_outcome_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["generic_outcome"])
        correct_choice = row["nl_context_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["generic_context"])
        incorrect_choice = row["nl_context_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["exception_context"])
        passed, _ = correctly_classify(tokenizer, model, prompt, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_four_prime(row):
        """
        experiment four: EXCEPTION OUTCOME -> EXCEPTION CONTEXT.
        The apple is [edible/<inedible>].
        (a) Jacob thinks the apple is delicious.
        (b) Jacob thinks the apple is rotten. <
        """
        prompt = row["nl_outcome_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["exception_outcome"])
        correct_choice = row["nl_context_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["exception_context"])
        incorrect_choice = row["nl_context_masked"].replace("[MASK]", tokenizer.mask_token).replace(tokenizer.mask_token, row["generic_context"])
        passed, _ = correctly_classify(tokenizer, model, prompt, correct_choice, incorrect_choice)
        return passed

    if model_name + "_pass_experiment_one_multiple_choice" not in new_df.columns:
        new_df[model_name + "_pass_experiment_one_multiplechoice"] = new_df.apply(lambda x: pass_experiment_one(x), axis=1)
    if model_name + "_pass_experiment_two_multiplechoice" not in new_df.columns:
        new_df[model_name + "_pass_experiment_two_multiplechoice"] = new_df.apply(lambda x: pass_experiment_two(x), axis=1)
    if model_name + "_pass_experiment_three_prime_multiplechoice" not in new_df.columns:
        new_df[model_name + "_pass_experiment_three_prime_multiplechoice"] = new_df.apply(lambda x: pass_experiment_three_prime(x), axis=1)
    if model_name + "_pass_experiment_four_prime_multiplechoice" not in new_df.columns:
        new_df[model_name + "_pass_experiment_four_prime_multiplechoice"] = new_df.apply(lambda x: pass_experiment_four_prime(x), axis=1)
    print("new df: ", new_df)
    return new_df


def add_pass_experiment_doubleheads(tokenizer, model, model_name, df, i, loadFromFile=False):
    new_df = df.copy()
    if loadFromFile: return new_df

    def pass_experiment_one(row):
        """
        experiment one: GENERIC CONTEXT -> GENERIC OUTCOME
        - [Jacob thinks the apple is [<delicious>]. The apple is edible. <<,
            Jacob thinks the apple is [<delicious>]. The apple is inedible.]
        """
        prompt = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        correct_choice = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        incorrect_choice = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        correct_choice = prompt + " " + correct_choice
        incorrect_choice = prompt + " " + incorrect_choice
        passed, _ = lm_choose_correctly(tokenizer, model, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_two(row):
        """
        experiment two: EXCEPTION CONTEXT -> EXCEPTION OUTCOME.
        - [Jacob thinks the apple is [delicious/<rotten>]. The apple is edible.,
            Jacob thinks the apple is [delicious/<rotten>]. The apple is inedible., <<]
        """
        prompt = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        correct_choice = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        incorrect_choice = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        correct_choice = prompt + " " + correct_choice
        incorrect_choice = prompt + " " + incorrect_choice
        passed, _ = lm_choose_correctly(tokenizer, model, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_three(row):
        """
        experiment three: GENERIC OUTCOME -> GENERIC CONTEXT.
        """
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        correct_choice = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        incorrect_choice = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        correct_choice = correct_choice + " " + outcome
        incorrect_choice = incorrect_choice + " " + outcome
        passed, _ = lm_choose_correctly(tokenizer, model, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_four(row):
        """
        experiment four: EXCEPTION OUTCOME -> EXCEPTION CONTEXT.
        """
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        correct_choice = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        incorrect_choice = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        correct_choice = correct_choice + " " + outcome
        incorrect_choice = incorrect_choice + " " + outcome
        passed, _ = lm_choose_correctly(tokenizer, model, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_three_prime(row):
        """
        experiment three prime: GENERIC OUTCOME -> GENERIC CONTEXT.
        """
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        correct_choice = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        incorrect_choice = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        correct_choice = outcome + " " + correct_choice
        incorrect_choice = outcome + " " + incorrect_choice
        passed, _ = lm_choose_correctly(tokenizer, model, correct_choice, incorrect_choice)
        return passed

    def pass_experiment_four_prime(row):
        """
        experiment four prime: EXCEPTION OUTCOME -> EXCEPTION CONTEXT.
        """
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        correct_choice = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        incorrect_choice = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        correct_choice = outcome + " " + correct_choice
        incorrect_choice = outcome + " " + incorrect_choice
        passed, _ = lm_choose_correctly(tokenizer, model, correct_choice, incorrect_choice)
        return passed

    if model_name + "_pass_experiment_one_doubleheads" not in new_df.columns:
        new_df[model_name + "_pass_experiment_one_doubleheads"] = new_df.apply(lambda x: pass_experiment_one(x), axis=1)
    if model_name + "_pass_experiment_two_doubleheads" not in new_df.columns:
        new_df[model_name + "_pass_experiment_two_doubleheads"] = new_df.apply(lambda x: pass_experiment_two(x), axis=1)

    if model_name + "_pass_experiment_three_doubleheads" not in new_df.columns:
        new_df[model_name + "_pass_experiment_three_doubleheads"] = new_df.apply(lambda x: pass_experiment_three(x), axis=1)
    if model_name + "_pass_experiment_four_doubleheads" not in new_df.columns:
        new_df[model_name + "_pass_experiment_four_doubleheads"] = new_df.apply(lambda x: pass_experiment_four(x), axis=1)

    if model_name + "_pass_experiment_three_prime_doubleheads" not in new_df.columns:
        new_df[model_name + "_pass_experiment_three_prime_doubleheads"] = new_df.apply(lambda x: pass_experiment_three_prime(x), axis=1)
    if model_name + "_pass_experiment_four_prime_doubleheads" not in new_df.columns:
        new_df[model_name + "_pass_experiment_four_prime_doubleheads"] = new_df.apply(lambda x: pass_experiment_four_prime(x), axis=1)
    
    print("new df, ahhhhhhhh: ", new_df)
    return new_df


def add_top_five_predictions_maskedlm(tokenizer, model, model_name, df, i, loadFromFile=False):
    def get_top_five(masked_prompt):
        top_five = one_mask_generation(tokenizer, model, masked_prompt, num_select=5, masked_sentence=masked_prompt)
        return ",".join(top_five)

    new_df = df.copy()
    if loadFromFile:
        return new_df

    print("Model name: ", model_name)

    def top_five_experiment_one(row):
        """
        experiment one: GENERIC CONTEXT -> GENERIC OUTCOME
        - Jacob thinks the apple is [<delicious>/rotten]. The apple is [MASK].
        """
        context = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        prompt = context + " " + row["nl_outcome_masked"]
        return get_top_five(prompt)

    def top_five_experiment_one_prime(row):
        """
        experiment one: GENERIC CONTEXT -> GENERIC OUTCOME
        - The apple is [MASK]. Jacob thinks the apple is [<delicious>/rotten].
        """
        context = row["nl_context_masked"].replace("[MASK]", row["generic_context"])
        prompt = row["nl_outcome_masked"] + " " + context
        return get_top_five(prompt)

    def top_five_experiment_two(row):
        """
        experiment two: EXCEPTION CONTEXT -> EXCEPTION OUTCOME
        - Jacob thinks the apple is [delicious/<rotten>]. The apple is [MASK].
        """
        context = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        prompt = context + " " + row["nl_outcome_masked"]
        return get_top_five(prompt)

    def top_five_experiment_two_prime(row):
        """
        experiment two: EXCEPTION CONTEXT -> EXCEPTION OUTCOME
        - The apple is [MASK]. Jacob thinks the apple is [delicious/<rotten>].
        """
        context = row["nl_context_masked"].replace("[MASK]", row["exception_context"])
        prompt = row["nl_outcome_masked"] + " " + context
        return get_top_five(prompt)

    def top_five_experiment_three(row):
        """
        experiment three: GENERIC OUTCOME -> GENERIC CONTEXT
        - Jacob thinks the apple is [MASK]. The apple is [<edible>/inedible].
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        prompt = context + " " + outcome
        return get_top_five(prompt)

    def top_five_experiment_four(row):
        """
        experiment four: EXCEPTION OUTCOME -> EXCEPTION CONTEXT
        - Jacob thinks the apple is [MASK]. The apple is [edible/<inedible>]
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        prompt = context + " " + outcome
        return get_top_five(prompt)

    def top_five_experiment_three_prime(row):
        """
        experiment three: GENERIC OUTCOME -> GENERIC CONTEXT
        - The apple is [<edible>/inedible]. Jacob thinks the apple is [MASK].
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["generic_outcome"])
        prompt = outcome + " " + context
        return get_top_five(prompt)

    def top_five_experiment_four_prime(row):
        """
        experiment four: EXCEPTION OUTCOME -> EXCEPTION CONTEXT
        - The apple is [edible/<inedible>]. Jacob thinks the apple is [MASK].
        """
        context = row["nl_context_masked"]
        outcome = row["nl_outcome_masked"].replace("[MASK]", row["exception_outcome"])
        prompt = context + " " + outcome
        return get_top_five(prompt)
    
    if model_name + "_topfive_experiment_one_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_one_maskedlm"] = new_df.apply(lambda x: top_five_experiment_one(x), axis=1)
    if model_name + "_topfive_experiment_one_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_one_prime_maskedlm"] = new_df.apply(lambda x: top_five_experiment_one_prime(x), axis=1)
    if model_name + "_topfive_experiment_two_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_two_maskedlm"] = new_df.apply(lambda x: top_five_experiment_two(x), axis=1)
    if model_name + "_topfive_experiment_two_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_two_prime_maskedlm"] = new_df.apply(lambda x: top_five_experiment_two_prime(x), axis=1)
    if model_name + "_topfive_experiment_three_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_three_maskedlm"] = new_df.apply(lambda x: top_five_experiment_three(x), axis=1)
    if model_name + "_topfive_experiment_four_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_four_maskedlm"] = new_df.apply(lambda x: top_five_experiment_four(x), axis=1)
    if model_name + "_topfive_experiment_three_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_three_prime_maskedlm"] = new_df.apply(lambda x: top_five_experiment_three_prime(x), axis=1)
    if model_name + "_topfive_experiment_four_prime_maskedlm" not in new_df.columns:
        new_df[model_name + "_topfive_experiment_four_prime_maskedlm"] = new_df.apply(lambda x: top_five_experiment_four_prime(x), axis=1)
    
    print("new_df: ", new_df)
    return new_df


if __name__ == "__main__":
    loaded_from_existing_files = [load_from_file(f, i) for i, f in enumerate(DFS)]
    # This is to add the context probability stuff
    count = 0
    for m_name in TOKENIZER_MODEL_DICT:
        tknzr, mdl = TOKENIZER_MODEL_DICT[m_name]
        if count == 0:
            added_probabilities = [add_probabilities(tknzr, mdl, m_name, f, i) for i, f in enumerate(loaded_from_existing_files)]
            count += 1
        else:
            added_probabilities = [add_probabilities(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_probabilities)]
    added_pos = [add_pos(f, i) for i, f in enumerate(added_probabilities)]
    added_nl_outcome = [add_nl_outcome(f, i) for i, f in enumerate(added_pos)]
    # This is to do the maskedlm information stuff
    count = 0
    for m_name in TOKENIZER_MODEL_DICT:
        tknzr, mdl = TOKENIZER_MODEL_DICT[m_name]
        if count == 0:
            added_experiment_masklm_per_model = [add_pass_experiment_maskedlm(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_nl_outcome)]
            count += 1
        else:
            added_experiment_masklm_per_model = [add_pass_experiment_maskedlm(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_masklm_per_model)]

    # This is to do the maskedlm top 5 generation stuff
    count = 0
    for m_name in TOKENIZER_MODEL_DICT:
        tknzr, mdl = TOKENIZER_MODEL_DICT[m_name]
        if count == 0:
            added_experiment_masklm_top_five_per_model = [add_top_five_predictions_maskedlm(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_masklm_per_model)]
            count += 1
        else:
            added_experiment_masklm_top_five_per_model = [add_top_five_predictions_maskedlm(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_masklm_top_five_per_model)]
    
    # # This is to do the multiplechoice information stuff
    # count = 0
    # for m_name in MULTIPLECHOICE_TOKENIZER_MODEL_DICT:
    #     tknzr, mdl = MULTIPLECHOICE_TOKENIZER_MODEL_DICT[m_name]
    #     if count == 0:
    #         added_experiment_multiplechoice_per_model = [add_pass_experiment_multiplechoice(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_masklm_per_model)]
    #         count += 1
    #     else:
    #         added_experiment_multiplechoice_per_model = [add_pass_experiment_multiplechoice(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_multiplechoice_per_model)]

    # # This is to do the doubleheads lm stuff:
    # count = 0
    # for m_name in DOUBLEHEADS_TOKENIZER_MODEL_DICT:
    #     tknzr, mdl = DOUBLEHEADS_TOKENIZER_MODEL_DICT[m_name]
    #     if count == 0:
    #         added_experiment_doubleheads_per_model = [add_pass_experiment_doubleheads(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_multiplechoice_per_model)]
    #         count += 1
    #     else:
    #         added_experiment_doubleheads_per_model = [add_pass_experiment_doubleheads(tknzr, mdl, m_name, f, i) for i, f in enumerate(added_experiment_doubleheads_per_model)]

    record_dfs(added_experiment_masklm_top_five_per_model, PROBABILITY_PATHS)
    # record_dfs(added_experiment_doubleheads_per_model, PROBABILITY_PATHS)