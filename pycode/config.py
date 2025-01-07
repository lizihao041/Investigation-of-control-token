# Description: Configuration file for the project
MODE = 'joint'
DATASET_DIR = 'dataset/Wikilarge/'
MODEL_OUTPUT_DIR = f'model/bart_base_{MODE}/'
MODEL_TO_LOAD = 'facebook/bart-base'
TOKENIZER_TO_LOAD = 'facebook/bart-base'
RESULT_DIR = 'result/'
STRATEGY = 'find_best_params'
TEST_SET = 'asset_valid'


#---------------Constants--------------------
PREPROCESSOR_NAMES = ['LengthRatioPreprocessor', 'ReplaceOnlyLevenshteinPreprocessor', 'WordRankRatioPreprocessor', 'DependencyTreeDepthRatioPreprocessor']
SUFFIX_NAMES = ['.txt','_LR.txt', '_RL.txt', '_WR.txt', '_DTD.txt']
PHASE = ['train','valid','test']
CONTROL_TOKENS = ['<DEPENDENCYTREEDEPTHRATIO_', '<WORDRANKRATIO_', '<REPLACEONLYLEVENSHTEIN_', '<LENGTHRATIO_']
SPECIAL_TOKEN_REGEX = r'<[a-zA-Z\-_\d\.]+>'
FASTTEXT_EMBEDDINGS_DIR = './dataset/fasttext_vectors'
VALUES = [str(round(0.05*x+0.2, 2)) for x in range(0, 27)]
max_input_length = 128
max_target_length = 128