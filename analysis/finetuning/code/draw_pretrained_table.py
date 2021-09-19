"""
TODO: DELETE THIS AS THIS IS NOT INTEGRAL TO THE MAIN CODE BASE
"""

import pandas as pd
import json, os

sequence = ["bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base",\
            "distilbert-base-cased", "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",\
                "allenai/longformer-large-4096", "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2"]

list_of_rows = []
for m in sequence:
    converted = m.replace("/", "-")
    if not os.path.exists(f"./{converted}-pretrained.json"):
        continue
    with open(f"{converted}-pretrained.json") as openfile:
        data = json.load(openfile)
    test_one_dict, test_two_dict = data["test_one"], data["test_two"]
    new_row = [m, test_one_dict["none"]*100, test_two_dict["none"]*100, test_one_dict["model_specific"]*100, test_two_dict["model_specific"]*100, test_one_dict["all_models"]*100, test_two_dict["all_models"]*100]
    list_of_rows.append(new_row)

columns = ["model", "all_generic", "all_exception", "indi_generic", "indi_exception", "intersect_generic", "intersect_exception"]
return_table = pd.DataFrame(list_of_rows, columns=columns)
print(return_table)
return_table.to_csv("pretrained_results.tsv", sep="\t")