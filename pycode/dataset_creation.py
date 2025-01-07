from datasets import Dataset, DatasetDict
from config import DATASET_DIR,max_input_length, max_target_length, PHASE


def creat_dataset(task):
    complex_list = []
    simple_list = []
    if task in PHASE:
        with open(DATASET_DIR + task + '_complex_DTD.txt') as complex:
            for i in complex.readlines():
                complex_list.append(i.strip())
        with open(DATASET_DIR + task + '_simple_DTD.txt') as simple:
            for i in simple.readlines():
                simple_list.append(i.strip())
        if len(complex_list) == len(simple_list):
            return Dataset.from_dict({'inputs': complex_list, 'targets': simple_list})
    elif task == 'sample':
        with open(DATASET_DIR + 'test_sample') as sample:
            for i in sample:
                complex_list.append(i.strip())
                simple_list.append('N/A')
        return Dataset.from_dict({'inputs': complex_list, 'targets': simple_list})
    else:
        raise TypeError('Invalid input!')

def creat_dataset_dict():
    return DatasetDict({'train': creat_dataset('train'), 'valid': creat_dataset('valid'), 'test': creat_dataset('test')})

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples['inputs'], max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['targets'], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs