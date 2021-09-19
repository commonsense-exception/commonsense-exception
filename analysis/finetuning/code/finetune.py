from transformers import BertTokenizer
import torch

# All the MaskedLM
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM,\
                            DistilBertTokenizer, DistilBertForMaskedLM, SqueezeBertTokenizer,\
                            SqueezeBertForMaskedLM, MobileBertTokenizer, MobileBertForMaskedLM,\
                            LongformerTokenizer, LongformerForMaskedLM, AlbertTokenizer,\
                            AlbertForMaskedLM

# other stuff
import subprocess, argparse, os, sys

# done: "bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base"
# "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",
# 
MODELS = [
    "bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base",\
    "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",\
    "allenai/longformer-large-4096", "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2"
]


############################## HELPERS ##############################

def parse_args():
    parser = argparse.ArgumentParser(description='finetuning BERT')
    parser.add_argument('-m', help="model name. {}".format(MODELS), default='bert-base-cased')
    parser.add_argument('-o', help="path to output the trained weights", default="output")
    parser.add_argument('-t', help="finetune type: ['winoventi/both_generic_exception', 'winoventi/only_exception', 'mnli', 'snli']")
    parser.add_argument('-fromprev', help="which checkpoint we're doing it from - 160000, 165000, etc.")
    # finetuning args
    parser.add_argument('-num_train_epochs', help="number of finetuning epochs", default=1500)
    parser.add_argument('-dept', help="whether we're on the department machine or not", default="True")
    return parser.parse_args()


def get_model_type(model_name):
    if model_name not in MODELS:
        raise Exception("invalid model name. Correct model names: {}".format(MODELS))
    model_name_to_type = {
        "bert-base-cased": "bert",
        "bert-large-cased-whole-word-masking": "bert",
        "roberta-base": "roberta",
        "roberta-large": "roberta",
        "distilbert-base-cased": "bert", # doubtful?
        "distilroberta-base": "roberta",
        "distilbert-base-uncased": "distilbert",
        "distilbert-base-uncased-distilled-squad": "distilbert",
        "squeezebert/squeezebert-uncased": "squeezebert",# ?
        "google/mobilebert-uncased": "mobilebert", # ?
        "allenai/longformer-base-4096": "longformer",
        "allenai/longformer-large-4096": "longformer",
        "albert-base-v2": "albert",
        "albert-large-v2": "albert",
        "albert-xlarge-v2": "albert",
        "albert-xxlarge-v2": "albert"
    }
    return model_name_to_type[model_name]


def get_correct_paths(model_name, dept=True, finetune_type="winoventi/both_generic_exception"):

    dept_prefix = "/data/nlp/ndo3/winoventi-commonsense-exception"
    cur_dir = os.getcwd()
    run_mlm_path = cur_dir + "/run_mlm.py"

    def get_correct_paths():    
        if dept:
            train_file_path = dept_prefix + "/data/finetune/{}/train.txt".format(finetune_type)
            validation_file_path = dept_prefix + "/data/finetune/{}/test.txt".format(finetune_type)
            output_dir_path = dept_prefix + "/code/finetuning/output/{}/{}".format(finetune_type, model_name)
            cache_dir_path = dept_prefix + "/code/finetuning/cache/{}".format(finetune_type)
        else:
            train_file_path = cur_dir + "/../../data/finetune/{}/train.txt".format(finetune_type)
            validation_file_path = cur_dir + "/../../data/finetune/{}/test.txt".format(finetune_type)
            output_dir_path = cur_dir + "/code/finetuning/output/{}/{}".format(finetune_type, model_name)
            cache_dir_path = "cache/{}".format(finetune_type)
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path, 0o770)
        if not os.path.exists(cache_dir_path):
            os.mkdir(cache_dir_path, 0o770)
        # assert all paths available
        assert os.path.exists(run_mlm_path)
        assert os.path.exists(train_file_path)
        assert os.path.exists(validation_file_path)
        assert os.path.exists(output_dir_path)
        assert os.path.exists(cache_dir_path)
        return run_mlm_path, train_file_path, validation_file_path, output_dir_path, cache_dir_path

    return get_correct_paths()
    


def main():
    args = parse_args()
    model_name, train_epochs = args.m, int(args.num_train_epochs)
    from_previous_checkpoint = args.fromprev
    finetune_type = args.t
    assert finetune_type in ["snli", "mnli", "winoventi/both_generic_exception", "winoventi/only_exception"]
    if args.dept == "True": dept_machine = True
    else: dept_machine = False
    run_mlm_path, train_file_path, validation_file_path, output_dir_path, cache_dir_path = get_correct_paths(model_name, \
                                                                                            dept=dept_machine, finetune_type=finetune_type)

    available_checkpoints = [int(e.replace("checkpoint-", "")) for e in os.listdir(output_dir_path) if "checkpoint-" in e]

    if from_previous_checkpoint != None:
        checkpoint_name = f"checkpoint-{from_previous_checkpoint}"
        checkpoint_path = f"{output_dir_path}/{checkpoint_name}"
        print("Requesting training from checkpoint: ", checkpoint_path)
        ret = subprocess.call([
            'python3', run_mlm_path, "--model_type={}".format(get_model_type(model_name)),\
                                    "--model_name_or_path={}".format(checkpoint_path),\
                                    "--train_file={}".format(train_file_path),\
                                    "--validation_file={}".format(validation_file_path),\
                                    "--line_by_line=True",\
                                    "--output_dir={}".format(output_dir_path),\
                                    "--cache_dir={}".format(cache_dir_path),\
                                    "--num_train_epochs={}".format(train_epochs),\
                                    "--do_train", "--do_eval", "--save_steps=5000", "--logging_steps=5000", "--overwrite_output_dir"
        ])
    elif len(available_checkpoints) == 0:
        print("Training from scratch!")
        ret = subprocess.call([
            'python3', run_mlm_path, "--model_type={}".format(get_model_type(model_name)),\
                                    "--model_name_or_path={}".format(model_name),\
                                    "--train_file={}".format(train_file_path),\
                                    "--validation_file={}".format(validation_file_path),\
                                    "--line_by_line=True",\
                                    "--output_dir={}".format(output_dir_path),\
                                    "--cache_dir={}".format(cache_dir_path),\
                                    "--num_train_epochs={}".format(train_epochs),\
                                    "--do_train", "--do_eval", "--save_steps=5000", "--logging_steps=5000", "--overwrite_output_dir"
        ])
    else:
        last_checkpoint = max(available_checkpoints)
        last_checkpoint = f"checkpoint-{last_checkpoint}"
        checkpoint_path = f"{output_dir_path}/{last_checkpoint}"
        print("Requesting training from checkpoint: ", checkpoint_path)
        ret = subprocess.call([
            'python3', run_mlm_path, "--model_type={}".format(get_model_type(model_name)),\
                                    "--model_name_or_path={}".format(checkpoint_path),\
                                    "--train_file={}".format(train_file_path),\
                                    "--validation_file={}".format(validation_file_path),\
                                    "--line_by_line=True",\
                                    "--output_dir={}".format(output_dir_path),\
                                    "--cache_dir={}".format(cache_dir_path),\
                                    "--num_train_epochs={}".format(train_epochs),\
                                    "--do_train", "--do_eval", "--save_steps=5000", "--logging_steps=5000", "--overwrite_output_dir"
        ])

    if ret < 0: print("Killed by signal", ret, -ret)
    else:
        if ret != 0: print("Command failed with return code: ", ret)
        else: print("subprocess call successfully completed?")


if __name__ == "__main__":
    main()
