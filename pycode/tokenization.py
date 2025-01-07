from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import VALUES, MODEL_OUTPUT_DIR, TOKENIZER_TO_LOAD, MODEL_TO_LOAD

def update_tokenizer(tokenizer, model, control_tokens, strategy):
    if strategy == 'joint':
        for i in control_tokens:
            if i == '<REPLACEONLYLEVENSHTEIN_':
                for j in VALUES[:-10]:
                    tokenizer.add_tokens(i + j + '>')
            else:
                for j in VALUES:
                    tokenizer.add_tokens(i + j + '>')
    elif strategy == 'sepa':
        if len(control_tokens) == 1 and control_tokens[0] == '<REPLACEONLYLEVENSHTEIN_':
            tokenizer.add_tokens(control_tokens)
            tokenizer.add_tokens(VALUES[:-10])
        else:
            tokenizer.add_tokens(control_tokens)
            tokenizer.add_tokens(VALUES)
    else:
        raise TypeError('Invalid input!')
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print('Tokenizer updated and the vocabulary size is : ' + str(len(tokenizer)))

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TO_LOAD)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TO_LOAD)
    return tokenizer, model