import pandas as pd
import argparse
from tqdm import tqdm

"""
The goal of this file is to produce jaccard analysis to analyze to what extent the
k-associative bias of one model coincides with that of another - across the different
model pairs and different Ks
"""

def parse_args():
    parser = argparse.ArgumentParser(description='arguments for analyzing the top-k bias generation')
    parser.add_argument('-top', help="Size of intersection of interest", default=10)
    parser.add_argument('-d', help="source data path (bias generation)",\
                                default="../../data/assets/k_analysis/")
    parser.add_argument('-o', help="output path to get the cross jaccard statistics",\
                                default="./assets/")
    
    # Brown University argument
    parser.add_argument('-dept', help="whether we're on the department machine or not", default="True")
    return parser.parse_args()
    
# Name of the models that we're dealing with
MODEL_NAMES = ["BERT_base", "BERT_large", "RoBERTa_small",\
                "RoBERTa_large", "DistilRoBERTa", "DistilBERT",\
                "SqueezeBERT", "MobileBERT", "Longformer_base",\
                "Longformer_large", "ALBERT_base", "ALBERT_large",\
                "ALBERT_xlarge", "ALBERT_xxlarge"]

# different values of K
Ks = [1, 3, 5, 8, 10]

def jaccard_index(input_one, input_two):
    """
    Function to get jaccard index between two sets separated by commas
    
    input_one: str, of format "a,b,c"
    input_two: str, of format "a,b,c"
    """
    if pd.isnull(input_one): input_one = ""
    if pd.isnull(input_two): input_two = ""
    set_one = set(input_one.split(","))
    set_two = set(input_two.split(","))
    return len(set_one.intersection(set_two)) / len(set_one.union(set_two))

def get_cross_jaccard_index(df, jaccard_outp_path):
    """
    Draws a table where each row and column corresponds to a 
    """
    def record_row(row, col_one_name, col_two_name, mutating_dict):
        val_one, val_two = row[col_one_name], row[col_two_name]
        mutating_dict["sum_jaccard"] += jaccard_index(val_one, val_two)
        mutating_dict["rows_recorded"] += 1

    # keeps track of models that we have completed jaccard analysis for, and
    # the results
    models_done = {}
    # for each pair of models
    for m_1 in MODEL_NAMES:
        for m_2 in MODEL_NAMES:
            # if the pair has yet to be computed, and are distinct
            if (m_2, m_1) in models_done or (m_1, m_2) in models_done: continue
            right_column_one, right_column_two = f"{m_1}_common", f"{m_2 }_common"
            models_dict = {"sum_jaccard": 0, "rows_recorded": 0}
            # record the jaccard statistics for this pair
            df.apply(lambda x: record_row(x, right_column_one, right_column_two, models_dict), axis=1)
            models_done[(m_1, m_2)] = models_dict

    # After we are done, then we will draw up the table
    list_of_rows = []
    for m_1 in MODEL_NAMES:
        new_row = [m_1]
        for m_2 in MODEL_NAMES:
            # get the stats for the pair
            pair = (m_1, m_2)
            if pair not in models_done: pair = (m_2, m_1)
            according_dict = models_done[pair]
            # Get the jaccard statistics
            average_jaccard = according_dict["sum_jaccard"] / according_dict["rows_recorded"]
            new_row.append(average_jaccard)
        
        # after we have considered all the possible other models, append this row to list of rows
        list_of_rows.append(new_row)
    
    # and then we will create a new pandas df
    column_names = ["models"]
    column_names.extend(MODEL_NAMES)
    final_df = pd.DataFrame(list_of_rows, columns=column_names)
    # final_df.to_csv(jaccard_outp_path, sep="\t", index=False)
    return final_df
            





if __name__ == "__main__":
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- JACCARD ANALYSIS -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    args = parse_args()
    for k in tqdm(Ks):
        correct_input_path = args.d + f"things_k{k}.tsv"
        correct_output_path = args.o + f"jaccard_k{k}.tsv"
        correct_df = pd.read_csv(correct_input_path, sep="\t")
        print(get_cross_jaccard_index(correct_df, correct_output_path))


        

