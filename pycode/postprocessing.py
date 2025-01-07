import os
import nevergrad as ng
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from config import MODEL_OUTPUT_DIR, RESULT_DIR, MODE
from preprocessors import get_preprocessor_by_name
from evaluation import evaluate_model
from utils import combine_metrics


def evaluate_parametrization(metrics_coefs, metrics, **preprocessors_kwargs):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
    simplifier = pipeline(task="summarization", model=model, tokenizer=tokenizer, device=0)
    scores = evaluate_model(simplifier, metrics, **preprocessors_kwargs)
    return combine_metrics(scores['bleu'], scores['sari'], scores['fkgl'], metrics_coefs)

def get_parametrization(preprocessors_kwargs):
    parametrization_kwargs = {}
    for preprocessor_name, preprocessor_kwargs in preprocessors_kwargs.items():
        assert '_' not in preprocessor_name
        parametrization_kwargs[preprocessor_name] = get_preprocessor_by_name(preprocessor_name).get_nevergrad_variables()
    return ng.p.Instrumentation(**parametrization_kwargs)

def find_best_parametrization(preprocessors_kwargs, metrics,metrics_coefs=[0, 1, 0], parametrization_budget=64):
    if os.path.exists(RESULT_DIR + 'score.txt'):
        os.remove(RESULT_DIR + 'score.txt')

    parametrization = get_parametrization(preprocessors_kwargs)
    if parametrization.dimension == 0:
        return preprocessors_kwargs

    parametrization_budget = min(32 * parametrization.dimension, parametrization_budget)
    optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=parametrization, budget=parametrization_budget, num_workers=1)
    optimizer.register_callback("tell", ng.callbacks.ProgressBar())
    recommendation = optimizer.minimize(lambda **kwargs: evaluate_parametrization(metrics_coefs, metrics, **kwargs), verbosity=0)
    return recommendation.kwargs