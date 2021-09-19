import pandas as pd
import numpy as np
import os, sys, random

SOURCE_PATH = "./../../data/final/winoventi_bert_large_final.tsv"
TRAIN_PATH = "./../../data/finetune/winoventi/train.txt"
TEST_PATH = "./../../data/finetune/winoventi/test.txt"

# check if train/test files exists already, if yes then we'll YEET
if os.path.exists(TRAIN_PATH) or os.path.exists(TEST_PATH):
    print("Train/Test already created. Exiting...")
    sys.exit()

# then, assert that the source path is there before we do stuff
assert os.path.exists(SOURCE_PATH)

# then, read the data, and do the magic!
data = pd.read_csv(SOURCE_PATH, sep="\t")
## First, combine the masked_prompt with the target to produce the training sentence
## That we're looking for
finetune_data = data.apply(lambda row: row["masked_prompt"].replace("[MASK]", row["target"]), axis=1)

list_data = np.array(data.values.tolist())
list_finetune_data = np.array(finetune_data.values.tolist())

## Selecting the indices to pull out as the train set
# training_indices = np.random.randint(0, int(len(data) / 2), size=int(len(data) / 4)).tolist()
training_indices = random.sample(range(0, int(len(data) / 2)), int(len(data) / 4))
other_training_indices = []
## Looping through the training indices to make sure that the other half (challenge 2) is the same shit
for i in training_indices:
    other_i = i + int(len(data) / 2)
    assert np.all(list_data[i][:5] == list_data[other_i][:5])
    other_training_indices.append(other_i)
training_indices.extend(other_training_indices)
training_indices = np.array(list(set(training_indices)))

## Getting testing indices
testing_indices = np.array([e for e in range(len(data)) if e not in training_indices])
## Testing that we've selected the correct train-test indices
assert len(training_indices) + len(testing_indices) == len(list_data)

list_finetune_train, list_finetune_test = list_finetune_data[training_indices], list_finetune_data[testing_indices]
train_df, test_df = pd.DataFrame(list_finetune_train), pd.DataFrame(list_finetune_test)

# alright. now output to the train and test path
train_df.to_csv(TRAIN_PATH, sep="\t", index=False, header=False)
test_df.to_csv(TEST_PATH, sep="\t", index=False, header=False)

print("Successfully outputted training samples to: ", TRAIN_PATH)
print("Successfully outputted testing samples to: ", TEST_PATH)
