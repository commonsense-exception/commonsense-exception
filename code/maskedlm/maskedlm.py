import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForMaskedLM

# init softmax
sm = torch.nn.Softmax(dim=0)
torch.set_grad_enabled(False)

MODEL_NAMES = ["BERT_base", "BERT_large", "RoBERTa_small",\
                "RoBERTa_large", "DistilRoBERTa", "DistilBERT",\
                "SqueezeBERT", "MobileBERT", "Longformer_base",\
                "Longformer_large", "ALBERT_base", "ALBERT_large",\
                "ALBERT_xlarge", "ALBERT_xxlarge"]

def load_model_and_tokenizer(model_name):
    """
    Input:
    - model_name: a string, that should be in the list of supported models
    """
    if model_name == "BERT_base":
        from transformers import BertTokenizer, BertForMaskedLM
        return (BertTokenizer.from_pretrained('bert-base-cased'), BertForMaskedLM.from_pretrained('bert-base-cased').eval())

    if model_name == "BERT_large":
        from transformers import BertTokenizer, BertForMaskedLM
        return (BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking'), BertForMaskedLM.from_pretrained('bert-large-cased-whole-word-masking').eval())

    if model_name == "RoBERTa_small":
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        return (RobertaTokenizer.from_pretrained('roberta-base'), RobertaForMaskedLM.from_pretrained('roberta-base').eval())

    if model_name == "RoBERTa_large":
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        return (RobertaTokenizer.from_pretrained('roberta-large'), RobertaForMaskedLM.from_pretrained('roberta-large').eval())
    
    if model_name == "DistilRoBERTa":
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        return (RobertaTokenizer.from_pretrained('distilroberta-base'), RobertaForMaskedLM.from_pretrained('distilroberta-base').eval())

    if model_name == "DistilBERT":
        from transformers import DistilBertTokenizer, DistilBertForMaskedLM
        return (DistilBertTokenizer.from_pretrained('distilbert-base-cased'), DistilBertForMaskedLM.from_pretrained('distilbert-base-cased').eval())

    if model_name == "SqueezeBERT":
        from transformers import SqueezeBertTokenizer, SqueezeBertForMaskedLM
        return (SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-uncased'), SqueezeBertForMaskedLM.from_pretrained('squeezebert/squeezebert-uncased').eval())

    if model_name == "MobileBERT":
        from transformers import MobileBertTokenizer, MobileBertForMaskedLM
        return (MobileBertTokenizer.from_pretrained('google/mobilebert-uncased'), MobileBertForMaskedLM.from_pretrained('google/mobilebert-uncased').eval())

    if model_name == "ALBERT_base":
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        return (AlbertTokenizer.from_pretrained('albert-base-v2'), AlbertForMaskedLM.from_pretrained('albert-base-v2').eval())

    if model_name == "ALBERT_large":
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        return (AlbertTokenizer.from_pretrained('albert-large-v2'), AlbertForMaskedLM.from_pretrained('albert-large-v2').eval())

    if model_name == "ALBERT_xlarge":
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        return (AlbertTokenizer.from_pretrained('albert-xlarge-v2'), AlbertForMaskedLM.from_pretrained('albert-xlarge-v2').eval())

    if model_name == "ALBERT_xxlarge":
        from transformers import AlbertTokenizer, AlbertForMaskedLM
        return (AlbertTokenizer.from_pretrained('albert-xxlarge-v2'), AlbertForMaskedLM.from_pretrained('albert-xxlarge-v2').eval())

    raise Exception("load_model_and_tokenizer was passed a model_name that is not supported")
    
def add_mask(sentence, mask_token, depth=1, ending="."):
    """
    Helper function to add masks to the end of a sentence. By default and
    in the paper, we only adding one mas to the end of the descriptor sentence.
    """
    begin = sentence.strip()
    for i in range(depth):
        begin += " [MASK]"
    begin += ending

    return begin.replace("..", ending).replace("[MASK]", mask_token)    

def one_mask_generation(tokenizer, model, sentence, num_select=5, masked_sentence=None, ending="."):
    """
    Function to get the most likely word (post softmax) to fill in the end of the sentence.
    Example usage: one_mask_generation(BERT tokenizer, BERT model, "The apple is ")
    """
    if masked_sentence == None:
        masked_sentence = add_mask(sentence, tokenizer.mask_token, ending=ending)
    else:
        masked_sentence = masked_sentence.replace("[MASK]", tokenizer.mask_token)

    token_ids = torch.tensor([tokenizer.encode(masked_sentence)])
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
    output = model(token_ids)
    last_hidden_state = output[0].squeeze(0)
    mask_hidden_state = last_hidden_state[masked_position]
    probs = sm(mask_hidden_state)
    sorted_ids = torch.argsort(probs, descending=True)
    predicted_tokens=tokenizer.convert_ids_to_tokens(sorted_ids)
    predicted_tokens=[e for e in predicted_tokens if "#" not in e and len(e) > 1]
    return predicted_tokens[:num_select]

def predict_prefix_probability(tokenizer, model, prefix, prediction, masked_prefix=None):
    """
    Function to get the probability that a word is to fill in a masked position, given
    the prefix. Return 0 if caught an error (sorry horrible engineering)

    Examplee usage:
    - predict_prefix_probability(BERT tokenizer, BERT model, "The apple is", "red") = some number
    - predict_prefix_probability(BERT tokenizer, BERT model, "Barack [MASK] is my hero", "Obama") = 1
    """
    try:
        if masked_prefix == None:
            masked_prefix = add_mask(prefix, tokenizer.mask_token, depth=1)
        else:
            masked_prefix = masked_prefix.replace("[MASK]", tokenizer.mask_token)

        prediction_id = tokenizer.convert_tokens_to_ids(prediction) # token id of the prediction
        token_ids = torch.tensor([tokenizer.encode(masked_prefix)])
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
        output = model(token_ids)
        last_hidden_state = output[0].squeeze(0)
        mask_hidden_state = last_hidden_state[masked_position]
        probs = sm(mask_hidden_state)
        predicted = probs[prediction_id].item()
    
    except Exception as e:
        return 0
    
    return predicted


# Dictionary of bidirectional Masked LMs - (tokenizer, model)
TOKENIZER_MODEL_DICT = {
    # replaced by load_model_and_tokenizer
}