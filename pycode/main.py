from preprocessing import preprocess_dataset
from training import train_model
from postprocessing import find_best_parametrization
from config import STRATEGY
from evaluation import evaluate_model_with_params
import torch

find_best_parametrization_args = {
        'preprocessors_kwargs': {
            'LengthRatioPreprocessor': {'target_ratio': 0.8},
            'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': 0.8},
            'WordRankRatioPreprocessor': {'target_ratio': 0.8, 'language': 'en'},
            'DependencyTreeDepthRatioPreprocessor': {'target_ratio': 0.8, 'language': 'en'}
        },
        'metrics': ['sari', 'bleu', 'fkgl'],
        'metrics_coefs': [0, 1, 0],
        'parametrization_budget': 1,
    }



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # preprocess_dataset()
    # train_model()
    if STRATEGY == 'find_best_params':
        best_params = find_best_parametrization(**find_best_parametrization_args)
        results = evaluate_model_with_params(best_params, ['sari', 'bleu', 'fkgl'])
        print(f'The best found params for control tokens on the test test are {best_params} and the corresponding score is {results}')
    # elif STRATEGY == 'predict_best_params':
    #     train_predict_model()
    #     best_params = predict_best_params()

    

if __name__ == '__main__':
    main()