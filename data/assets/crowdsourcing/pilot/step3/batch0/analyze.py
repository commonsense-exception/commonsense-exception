import pandas as pd
from process import module_from_file, record_dfs, PROBABILITY_PATHS
import sys

stats = {}

#TOKENIZER_MODEL_DICT_KEYS = ["BERT", "RoBERTa", "DistilBERT", "SqueezeBERT", "MobileBERT", "Longformer", "FunnelTransformer", "ELECTRA", "ALBERT", "XLMRoBERTa"]
TOKENIZER_MODEL_DICT_KEYS = ["BERT_base", "BERT_large", "RoBERTa_small", "RoBERTa_large", "DistilRoBERTa", "DistilBERT", "SqueezeBERT", "MobileBERT", "Longformer_base", "Longformer_large", "ALBERT_base", "ALBERT_large", "ALBERT_xlarge", "ALBERT_xxlarge"]
MODEL_NAMES = sorted(TOKENIZER_MODEL_DICT_KEYS)
ANALYZED_DF_MASKEDLM_PATHS = ["batch0_0_analyzed_maskedlm.csv"]
ANALYZED_DF_MULTIPLECHOICE_PATHS = ["batch0_0_analyzed_multiplechoice.csv"]
ANLYZED_PROBABILITIES_PATHS = ["batch0_0_analyzed_probabilities.csv"]
assert len(ANALYZED_DF_MASKEDLM_PATHS) == len(PROBABILITY_PATHS)
assert len(ANALYZED_DF_MULTIPLECHOICE_PATHS) == len(PROBABILITY_PATHS)
# constants
maskedlm_experiment_names = ["pass_experiment_" + e + "_maskedlm" for e in ["one", "one_prime", "two_prime", "two", "three", "four", "three_prime", "four_prime"]]
multiplechoice_experiment_names = ["pass_experiment_" + e + "_multiplechoice" for e in ["one", "two", "three_prime", "four_prime"]]
multiplechoice_experiment_names = ["pass_experiment_" + e + "_doubleheads" for e in ["one", "two", "three", "four", "three_prime", "four_prime"]]
probability_experiment_names = ["context", "outcome"]



def create_empty_dataframe_with_modelname(df, i, test_type, loadFromFile=False):
    if loadFromFile:
        return pd.read_csv(ANALYZED_DF_MASKEDLM_PATHS[i], sep="\t")
    a = pd.DataFrame()
    a[test_type] = MODEL_NAMES.copy()
    return a


def get_experiment_stats(df, i, test_type, experiment_name, loadFromFile=False):
    if loadFromFile:
        return pd.read_csv(ANALYZED_DF_MASKEDLM_PATHS[i], sep="\t")
    
    def get_stats_given_model(raw_df, model_name):
        right_column_name = model_name + "_(EXPERIMENT_NAME)".replace("(EXPERIMENT_NAME)", experiment_name)
        # return accuracy
        return (raw_df[raw_df[right_column_name]].shape[0])/(raw_df.shape[0])

    df[experiment_name] = df.apply(lambda row: get_stats_given_model(pd.read_csv(PROBABILITY_PATHS[i], sep="\t"), row[test_type]), axis=1)
    return df


def get_probability_correct(df, i, test_type, experiment_name, loadFromFile=False):
    """
    test_type: "Probabilities"
    experiment name: 'context' or 'outcome'
    """
    if loadFromFile:
        return pd.read_csv(ANALYZED_DF_MULTIPLECHOICE_PATHS[i], sep="\t")

    print("input df: ", df)

    def get_probability_context_exception_smaller(raw_df, model_name):
        p_exception_column_name = "p_exception_(EXPERIMENT_NAME)_".replace("(EXPERIMENT_NAME)", experiment_name) + model_name
        p_generic_column_name = "p_generic_(EXPERIMENT_NAME)_".replace("(EXPERIMENT_NAME)", experiment_name) + model_name
        correct = raw_df[raw_df[p_exception_column_name] < raw_df[p_generic_column_name]]
        return (correct.shape[0])/(raw_df.shape[0])

    df[experiment_name] = df.apply(lambda row: get_probability_context_exception_smaller(pd.read_csv(PROBABILITY_PATHS[i], sep="\t"), row[test_type]), axis=1)
    return df

    


# MaskedLM ################################################
maskedlm_stats_dfs = [create_empty_dataframe_with_modelname(f, i, 'MaskedLM') for i, f in enumerate(PROBABILITY_PATHS)]
for experiment in maskedlm_experiment_names:
    maskedlm_stats_dfs = [get_experiment_stats(f, i, "MaskedLM", experiment) for i, f in enumerate(maskedlm_stats_dfs)]

# RECORD + PRINT OUT MASKEDLM STATS
record_dfs(maskedlm_stats_dfs, ANALYZED_DF_MASKEDLM_PATHS)
print(maskedlm_stats_dfs[0])

sys.exit()

# Multiple Choice ##########################################
multiplechoice_stats_dfs = [create_empty_dataframe_with_modelname(f, i, 'MultipleChoice') for i, f in enumerate(PROBABILITY_PATHS)]
for experiment in multiplechoice_experiment_names:
    multiplechoice_stats_dfs = [get_experiment_stats(f, i, "MultipleChoice", experiment) for i, f in enumerate(multiplechoice_stats_dfs)]

# RECORD + PRINT OUT MULTIPLECHOICE STATS 
record_dfs(multiplechoice_stats_dfs, ANALYZED_DF_MULTIPLECHOICE_PATHS)
print(multiplechoice_stats_dfs)

# the probability of context ################################
probability_dfs = [create_empty_dataframe_with_modelname(f, i, 'Probabilities') for i, f in enumerate(PROBABILITY_PATHS)]
for experiment in probability_experiment_names:
    probability_dfs = [get_probability_correct(f, i, "Probabilities", experiment) for i, f in enumerate(probability_dfs)]

record_dfs(probability_dfs, ANLYZED_PROBABILITIES_PATHS)
print(probability_dfs)


# DOUBLEHEAD MODELS ##########################################
doubleheads_stats_dfs = [create_empty_dataframe_with_modelname(f, i, 'DoubleHeads') for i, f in enumerate(PROBABILITY_PATHS)]
for experiment in multiplechoice_experiment_names:
    doubleheads_stats_dfs = [get_experiment_stats(f, i, "DoubleHeads", experiment) for i, f in enumerate(doubleheads_stats_dfs)]