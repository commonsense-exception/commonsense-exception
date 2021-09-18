import pandas as pd
import sys, argparse
import importlib.util
from tqdm import tqdm

NUM_SELECTING = 10

def parse_args():
    parser = argparse.ArgumentParser(description='arguments for generating biases')
    parser.add_argument('-top', help="Size of intersection of interest", default=10)
    parser.add_argument('-d', help="source data path (THINGS dataset)",\
                                default="../../data/source/things_concepts.tsv")
    parser.add_argument('-o', help="output path - path to the top-k associative biases",\
                                default="../../data/assets/k_analysis/")
    
    # Brown University argument
    parser.add_argument('-dept', help="whether we're on the department machine or not", default="True")
    return parser.parse_args()


# helper function to help load things from maskedlm folder
def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

MASKEDLM_MODULE_PATH = "../maskedlm/maskedlm.py"

# importimng the function one_mask_generation from maskedlm.py
one_mask_generation = module_from_file('one_mask_generation', MASKEDLM_MODULE_PATH).one_mask_generation

# importing the list of supported models from maskedlm.py
MODEL_NAMES = module_from_file('one_mask_generation', MASKEDLM_MODULE_PATH).MODEL_NAMES

# import the function to load the tokenizer
load_model_and_tokenizer = module_from_file('load_model_and_tokenizer', MASKEDLM_MODULE_PATH).load_model_and_tokenizer

def find_common_given_model(tokenizer, model, model_name, df, loadFromFile=False, top=10):
    """
    input:
    - tokenizer: HuggingFace's tokenizer
    - model: HuggingFace MaskedLM models
    - model_name: str, name of model
    - df: pandas dataframe that keeps our information
    - loadFromFile: whether we're gonna skip this step or not
    """
    new_df = df.copy()
    if loadFromFile:
        return new_df

    def process_data_row(row, num_selecting=top):
        word = row["Word"]
        ############################## ENDING WITH A "." ###################################
        # use bert generation to select the top 10 words
        affirmative = "The (OBJECT) is".replace("(OBJECT)", word)
        negative = "The (OBJECT) is not".replace("(OBJECT)", word)
        # top k affirmative and negative
        top_ten_aff_dot = one_mask_generation(tokenizer, model, affirmative, num_selecting, ending=".")
        top_ten_neg_dot = one_mask_generation(tokenizer, model, negative, num_selecting, ending=".")
        # find the intersection
        common_dot = set(top_ten_aff_dot).intersection(set(top_ten_neg_dot))
        ############################## ENDING WITH A "," ###################################
        # top 10 affirmative and negative
        top_ten_aff_comma = one_mask_generation(tokenizer, model, affirmative, num_selecting, ending=",")
        top_ten_neg_comma = one_mask_generation(tokenizer, model, negative, num_selecting, ending=",")
        # find the intersection
        common_comma = set(top_ten_aff_comma).intersection(set(top_ten_neg_comma))
        ############################# UNION THEM TOGETHER ##################################
        common = common_dot.union(common_comma)
        return ",".join([e for e in common])
    
    if model_name + "_common" not in new_df.columns:
        new_df[model_name + "_common"] = new_df.apply(lambda x: process_data_row(x), axis=1)
    
    return new_df



if __name__ == "__main__":
    args = parse_args()
    DATA_PATH = args.d
    NUM_SELECTING = args.top
    OUT_PATH = args.o + f"things_k{NUM_SELECTING}.tsv"
    try:
        # if this works, it means that there has been a file here before, and we will load from it and we will go from there
        DATA = pd.read_csv(OUT_PATH, sep="\t")
    except Exception as e:
        # if the path is not valid, we'll load from source
        DATA = pd.read_csv(DATA_PATH, sep="\t")
        DATA = DATA[["Word"]]
    ## Loop cross models
    for m_name in tqdm(MODEL_NAMES):
        tknzr, mdl = load_model_and_tokenizer(m_name)
        DATA = find_common_given_model(tknzr, mdl, m_name, DATA, top=NUM_SELECTING)
        DATA.to_csv(OUT_PATH, index=False, sep="\t")


