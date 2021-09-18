#############################################  IMPORT STUFF  #############################################
import pandas as pd
import numpy as np
import importlib.util
from spellchecker import SpellChecker

# helper function to help load things from BERT folder
def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# load function from BERT to process things
MASKEDLM_MODULE_PATH = "../../../../code/BERT/bert_generation.py"

bert_generation = module_from_file('predict_prefix_probability', MASKEDLM_MODULE_PATH)
predict_prefix_probability = bert_generation.predict_prefix_probability
spell = SpellChecker()

##########################################  END OF IMPORT STUFF  ##########################################

FILES = ["batch_0_raw.csv"]
PROBABILITY_PATHS = ["batch_0_with_probs.csv"]
DFS = [pd.read_csv(f, sep=",") for f in FILES]
TEMPLATE = "The (WORD) is"

def clean_df(df, filterNonsense=True, filterNoncommonsense=False, lower=True, fixTypo=True, saveToPath=True):
    # first, filter out nonsense
    new_df = df.copy()
    log = "og: " + str(df.shape[0]) + ". "
    if filterNonsense:
        new_df = new_df[new_df["makeSense"] == "Yes"]
        log += "filtered nonsense, remaining: " + str(new_df.shape[0]) + ". "
    if filterNoncommonsense:
        new_df = new_df[new_df["frequentAssociation"]]
        log += "filtered non frequent association, remaining: " + str(new_df.shape[0]) + ". "
    cols = ["antonym1", "antonym2", "antonym3"]
    if lower:
        def lower_item(i):
            if pd.isnull(i): return ""
            return i.lower()
        for col in cols:
            new_df[col] = new_df[col].apply(lambda x: lower_item(x))
    if fixTypo:
        def fix_typo(i):
            if pd.isnull(i) or i == "": return ""
            return spell.correction(i)
        for col in cols:
            new_df[col] = new_df[col].apply(lambda x: fix_typo(x))
    return new_df, log


def add_probabilities(df, i, loadFromFile=True):
    if loadFromFile:
        return pd.read_csv(PROBABILITY_PATHS[i], sep="\t")
    new_df = df.copy()
    def find_probability(prefix, notion, adj):
        if pd.isnull(adj) or adj == "": return 0
        return predict_prefix_probability(prefix.replace("(WORD)", notion), adj)
    cols = ["generic", "antonym1", "antonym2", "antonym3"]
    for col in cols:
        new_df["p_" + col] = new_df.apply(lambda x: find_probability(TEMPLATE, x["Word"], x[col]), axis=1)
    # and then make a final column of the words that are applicable
    def applicable_adj(row):
        first, second, third = row[cols[1]], row[cols[2]], row[cols[3]]
        p_first, p_second, p_third = row["p_"+cols[1]], row["p_"+cols[2]], row["p_"+cols[3]]
        p_generic = row["p_generic"]
        results = []
        if p_first < p_generic: results.append(first)
        if p_second < p_generic: results.append(second)
        if p_third < p_generic: results.append(third)
        results = [e for e in results if e != ""]
        return ",".join(results)
    new_df["qualified"] = new_df.apply(lambda x: applicable_adj(x), axis=1)
    return new_df


def record_dfs(list_dfs, list_paths):
    assert len(list_dfs) == len(list_paths)
    for i, df in enumerate(list_dfs):
        df.to_csv(list_paths[i], index=False, sep="\t")


cleaned_dfs = [clean_df(f) for i, f in enumerate(DFS)]
print("Log: ", [log[1] for log in cleaned_dfs])
added_probabilities = [add_probabilities(f, i) for i, f in enumerate(cleaned_dfs)]
# print(added_probabilities[0])
record_dfs(added_probabilities, PROBABILITY_PATHS)