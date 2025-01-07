from preprocessors import get_preprocessors
import os
from typing import List
from config import PHASE, SUFFIX_NAMES, PREPROCESSOR_NAMES, DATASET_DIR

preprocessors_kwargs = {
    'LengthRatioPreprocessor': {'target_ratio': 0.8, 'use_short_name': False},
    'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': 0.8, 'use_short_name': False},
    'WordRankRatioPreprocessor': {'target_ratio': 0.8, 'language': 'en', 'use_short_name': False},
    'DependencyTreeDepthRatioPreprocessor': {'target_ratio': 0.8, 'language': 'en', 'use_short_name': False}
}

def create_preprocessed_dataset(preprocessors: List, dataset_dir: str,phase) -> None:
    """
    Create preprocessed datasets using the provided preprocessors.
    """
    dataset_paths = []
    for i in SUFFIX_NAMES:
        dataset_paths.append(f'{dataset_dir}{phase}_complex{i}')
        dataset_paths.append(f'{dataset_dir}{phase}_simple{i}')
    for preprocessor in preprocessors:
        create_preprocessed_dataset_one_preprocessor(preprocessor, dataset_paths)


def create_preprocessed_dataset_one_preprocessor(preprocessor,dataset_paths):
    preprocessor_name = preprocessor.get_hash_string().split('(')[0]
    if preprocessor_name in PREPROCESSOR_NAMES:
        index = PREPROCESSOR_NAMES.index(preprocessor_name)
        print('Appending the ' + preprocessor_name + ' for ' + dataset_paths[index*2])
        preprocessor.encode_file_pair(dataset_paths[index*2],dataset_paths[index*2+1],dataset_paths[index*2+2],dataset_paths[index*2+3])


def preprocess_dataset(dataset_dir=DATASET_DIR):
    if all(not file.endswith('_DTD.txt') for file in os.listdir(dataset_dir)):
        preprocessors = get_preprocessors(preprocessors_kwargs)
        for phase in PHASE:
            create_preprocessed_dataset(preprocessors,dataset_dir,phase)
    else:
        print('The dataset has already been preprocessed.')
        