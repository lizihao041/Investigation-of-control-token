from easse.cli import evaluate_system_output, get_orig_and_refs_sents
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from utils import write_lines
from config import RESULT_DIR, MODEL_OUTPUT_DIR, TEST_SET
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def evaluate_model(simplifier, orig_sents_path=None, refs_sents_paths=None, quality_estimation=False,metrics = ['sari', 'bleu', 'fkgl'], **preprocessors_kwargs):

    test_set = TEST_SET

    orig_sents, refs_sents = get_orig_and_refs_sents(
        test_set, orig_sents_path=orig_sents_path, refs_sents_paths=refs_sents_paths
    )
    
    orig_sents_path = RESULT_DIR + 'origin.txt'
    sys_sents_path = RESULT_DIR + 'system.txt'
    write_lines(orig_sents, orig_sents_path)
    

    preprocessors = '<DEPENDENCYTREEDEPTHRATIO_'  + str(preprocessors_kwargs.get('DependencyTreeDepthRatioPreprocessor')) + \
                    '> <WORDRANKRATIO_' + str(preprocessors_kwargs.get('WordRankRatioPreprocessor')) + \
                    '> <REPLACEONLYLEVENSHTEIN_' + str(preprocessors_kwargs.get('ReplaceOnlyLevenshteinPreprocessor')) + \
                    '> <LENGTHRATIO_' + str(preprocessors_kwargs.get('LengthRatioPreprocessor')) + \
                    '>'

    with open(sys_sents_path, 'w', encoding='utf-8') as s:
        prepro_sents = []
        with open(orig_sents_path,'r', encoding='utf-8') as o:
            orig_sents = o.readlines()
            for i in orig_sents:
                prepro_sents.append(preprocessors+str(i.strip()))
            test_dataset = Dataset.from_dict({'text':prepro_sents})
            for out in simplifier(KeyDataset(test_dataset, "text"), batch_size=32):
                s.write(out[0].get('summary_text')+ '\n')

    return evaluate_system_output(
        test_set,
        sys_sents_path=sys_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        metrics=metrics,
        quality_estimation=quality_estimation,
    )

def evaluate_model_with_params(best_params, metrics):

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
    simplifier = pipeline(task="summarization", model=model, tokenizer=tokenizer, device=0)

    return evaluate_model(simplifier, metrics = metrics, **best_params)