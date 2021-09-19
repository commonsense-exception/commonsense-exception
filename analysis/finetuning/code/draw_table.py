import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json

MODELS = [
    "bert-base-cased", "bert-large-cased-whole-word-masking", "roberta-base", "roberta-large", "distilroberta-base",\
    "squeezebert/squeezebert-uncased", "google/mobilebert-uncased", "allenai/longformer-base-4096",\
    "allenai/longformer-large-4096", "albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2"
]


def parse_args():
    parser = argparse.ArgumentParser(description='finetuning BERT')
    parser.add_argument('-m', help="model name: [bert-base-cased, roberta-base, etc.]")
    parser.add_argument('-finetunemode', help="finetune type: ['winoventi/both_generic_exception', 'winoventi/only_exception', 'mnli', 'snli']")#, default='winoventi/both_generic_exception')
    parser.add_argument('-afmode', help="adversarial filtering or not ['none', 'model_specific', 'all_models']",\
                                                    default='model_specific')
    return parser.parse_args()


def draw_graphs_both_generic_exception(model_name, afmode):
    model_name = model_name.replace("/", "-")
    finetunemode = "winoventi/both_generic_exception"
    data_path = f"./{finetunemode}/{model_name}-{afmode}.json"
    with open(data_path, "r") as openfile:
        data = json.load(openfile)
    checkpoints = list(sorted([int(e) for e in data if e != "checkpoint" and int(e) % 5000 == 0 and int(e) <= 190000]))
    print("checkpoints: ", checkpoints)
    # first, get the data
    accuracies_train_one = [data[str(e)]["accuracy_one_train"] for e in checkpoints]
    accuracies_test_one = [data[str(e)]["accuracy_one_test"] for e in checkpoints]
    accuracies_train_two = [data[str(e)]["accuracy_two_train"] for e in checkpoints]
    accuracies_test_two = [data[str(e)]["accuracy_two_test"] for e in checkpoints]
    # second, drawing the challenges one (generic)
    plt.plot(checkpoints, accuracies_train_one, label="Training accuracy")
    plt.plot(checkpoints, accuracies_test_one, label="Testing accuracy - generics")
    plt.plot(checkpoints, accuracies_test_two, label="Testing accuracy - exceptions")
    plt.xlabel("Checkpoints")
    plt.ylabel("Accuracy")
    plt.title(f"""Evaluation of {model_name} on testing challenges after\nfinetuning on both generic and exception training schemas""")
    

    plt.legend()
    plt.show()
    # third, drawing the challenges two (exception)
    # plt.plot(checkpoints, accuracies_train_two, label="Training accuracy")
    # plt.plot(checkpoints, accuracies_test_two, label="Testing accuracy")
    # plt.xlabel("Checkpoints")
    # plt.ylabel("Accuracy")
    # plt.title(f"Evaluation of {model_name} on exception challenges after finetuning on\nboth generic and exception schemas")
    
    # plt.legend()
    # plt.show()


def draw_graphs_only_exception(model_name, afmode):
    # figure(figsize=(8, 6), dpi=80)
    model_name = model_name.replace("/", "-")
    finetunemode = "winoventi/only_exception"
    data_path = f"./{finetunemode}/{model_name}-{afmode}.json"
    with open(data_path, "r") as openfile:
        data = json.load(openfile)
    checkpoints = list(sorted([int(e) for e in data if e != "checkpoint" and int(e) % 5000 == 0]))
    print("checkpoints: ", checkpoints)
    # first, get the data
    accuracies_train = [data[str(e)]["accuracy_train"] for e in checkpoints]
    accuracies_test_heldout_exception = [data[str(e)]["accuracy_test_heldout_exception"] for e in checkpoints]
    accuracies_test_all_generics = [data[str(e)]["accuracy_test_all_generics"] for e in checkpoints]
    accuracies_one_test_both_generic_exception = [data[str(e)]["accuracy_one_test_both_generic_exception"] for e in checkpoints]
    accuracies_two_test_both_generic_exception = [data[str(e)]["accuracy_two_test_both_generic_exception"] for e in checkpoints]
    # second, drawing the challenges (first: train exception and test exception)
    plt.plot(checkpoints, accuracies_train, label="Training accuracy")
    plt.plot(checkpoints, accuracies_test_heldout_exception, label="Testing accuracy")
    plt.xlabel("Checkpoints")
    plt.ylabel("Accuracy")
    plt.title(f"Evaluation of {model_name} on\ntraining and testing exception challenges")
    plt.legend()
    plt.show()
    # (second: train exception and test generics)
    plt.plot(checkpoints, accuracies_train, label="Training accuracy")
    plt.plot(checkpoints, accuracies_test_all_generics, label="Testing accuracy")
    plt.xlabel("Checkpoints")
    plt.ylabel("Accuracy")
    plt.title(f"Evaluation of {model_name} on training exception challenges\nand testing generic challenges")
    plt.legend()
    plt.show()
    # (third: train exception and accuracy on the both)
    plt.plot(checkpoints, accuracies_train, label="Training accuracy")
    plt.plot(checkpoints, accuracies_one_test_both_generic_exception, label="Testing accuracy - generics")
    plt.plot(checkpoints, accuracies_two_test_both_generic_exception, label="Testing accuracy - exceptions")
    plt.title(f"Evaluation of {model_name} on training challenges (exception)\nand test challenges (generics and exception)")
    plt.xlabel("Checkpoints")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    


    



def main():
    args = parse_args()
    model, ft_mode, af_mode = args.m, args.finetunemode, args.afmode
    if ft_mode == "winoventi/both_generic_exception":
        draw_graphs_both_generic_exception(model, af_mode)
    if ft_mode == "winoventi/only_exception":
        draw_graphs_only_exception(model, af_mode)
        

if __name__ == "__main__":
    main()
    
    