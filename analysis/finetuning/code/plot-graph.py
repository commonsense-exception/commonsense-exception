import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, argparse

MODELS = [
    "bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base",\
    "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",\
    "allenai/longformer-large-4096", "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2"
]

def parse_args():
    parser = argparse.ArgumentParser(description='plotting graph')
    parser.add_argument('-finetunemode', help="fine tune mode ['winoventi/both_generic_exception', 'winoventi/only_exception']"\
                                        , default='winoventi/both_generic_exception')
    parser.add_argument('-challenge', help="challenge mode: ['EXCEPTION', 'GENERIC', 'AVERAGE']", default='EXCEPTION')
    parser.add_argument('-accuracy', help="accuracy mode: ['TRAIN', 'TEST']", default='TEST')
    return parser.parse_args()

args = parse_args()

FINETUNE_MODE = args.finetunemode
challenge_mode = args.challenge
assert challenge_mode in ['EXCEPTION', 'GENERIC', 'AVERAGE']
accuracy_mode = args.accuracy
assert accuracy_mode in ['TRAIN', 'TEST']

####### VARIABLES TO CHANGE #######
xs = []
model_to_values = {}
caption = True

###################################

# other types of processing first
if challenge_mode == "EXCEPTION": challenge = "two" 
else: challenge = "one" 

for m in MODELS:
    supposed_path = "{}/{}-model_specific.tsv".format(FINETUNE_MODE, m)
    if not os.path.exists(supposed_path):
        continue
    data = pd.read_csv(supposed_path, sep="\t")
    if len(data) < 23:
        continue
    data = data[:24]
    if len(xs) == 0:
        xs = list(data["checkpoint"])
        model_to_values["x_values"] = xs
    
    if challenge_mode == "GENERIC" or challenge_mode == "EXCEPTION":
        correct_column_name = "accuracy_{}_{}".format(challenge, accuracy_mode.lower())
        model_to_values[m] = list(data[correct_column_name])
        print("accuracy_mode: ", accuracy_mode)
    else:
        generic_column_name = "accuracy_{}_{}".format("one", accuracy_mode.lower())
        exception_column_name = "accuracy_{}_{}".format("two", accuracy_mode.lower())
        generic_values = np.array(list(data[generic_column_name]))
        exception_values = np.array(list(data[exception_column_name]))
        average_values = (generic_values + exception_values) / 2
        model_to_values[m] = list(average_values)

# to draw pandas dataframe
to_draw_df = pd.DataFrame(model_to_values)
for k in to_draw_df:
    if k == "x_values": continue
    plt.plot("x_values", k, data=to_draw_df)

# show legend
plt.ylabel('Accuracy')
plt.xlabel('Checkpoints')
if caption:
    if challenge_mode != "AVERAGE":
        plt.title('Change in performance on {} challenges ({}) after finetuning'.format(challenge_mode.lower(), accuracy_mode.lower()))
    else:
        plt.title('Change in the average performance of generic and exception challenges \n ({}) after finetuning'.format(accuracy_mode.lower()))
plt.legend()

# show graph
plt.show()
    
    
    
