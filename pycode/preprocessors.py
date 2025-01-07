from abc import ABC
import hashlib
import numpy as np
import nevergrad as ng

# Utility functions
from utils import (
    store_args, get_temp_filepath,
    yield_lines_in_parallel, write_lines_in_parallel, remove_special_tokens,
    extract_special_tokens, failsafe_division, get_lexical_complexity_score,
    get_levenshtein_similarity, get_replace_only_levenshtein_similarity, get_dependency_tree_depth
)

PREPROCESSORS_REGISTRY = {}

def get_preprocessors(preprocessors_kwargs):
    preprocessors = []
    for preprocessor_name, kwargs in preprocessors_kwargs.items():
        preprocessors.append(get_preprocessor_by_name(preprocessor_name)(**kwargs))
    return preprocessors

def get_preprocessor_by_name(preprocessor_name):
    return PREPROCESSORS_REGISTRY[preprocessor_name]

class AbstractPreprocessor(ABC):
    def __init_subclass__(cls, **kwargs):
        '''Register all children in registry'''
        super().__init_subclass__(**kwargs)
        PREPROCESSORS_REGISTRY[cls.__name__] = cls

    def __repr__(self):
        args = getattr(self, 'args', ())
        kwargs = getattr(self, 'kwargs', {})
        args_repr = [repr(arg) for arg in args]
        kwargs_repr = [f'{k}={repr(v)}' for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])]
        args_kwargs_str = ', '.join(args_repr + kwargs_repr)
        return f'{self.__class__.__name__}({args_kwargs_str})'

    def get_hash_string(self):
        return self.__class__.__name__

    def get_hash(self):
        return hashlib.md5(self.get_hash_string().encode()).hexdigest()

    @staticmethod
    def get_nevergrad_variables():
        return {}

    @property
    def prefix(self):
        return self.__class__.__name__.replace('Preprocessor', '')

    def fit(self, complex_filepath, simple_filepath):
        pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def decode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        if complex_sentence is not None:
            complex_sentence = self.encode_sentence(complex_sentence)
        if simple_sentence is not None:
            simple_sentence = self.encode_sentence(simple_sentence)
        return complex_sentence, simple_sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for input_line, encoder_line in yield_lines_in_parallel([input_filepath, encoder_filepath], strict=False):
                f.write(self.encode_sentence(input_line, encoder_line) + '\n')

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for encoder_sentence, input_sentence in yield_lines_in_parallel(
                [encoder_filepath, input_filepath], strict=False
            ):
                decoded_sentence = self.decode_sentence(input_sentence, encoder_sentence=encoder_sentence)
                f.write(decoded_sentence + '\n')

    def encode_file_pair(self, complex_filepath, simple_filepath, output_complex_filepath, output_simple_filepath):
        with write_lines_in_parallel([output_complex_filepath, output_simple_filepath], strict=False) as output_files:
            for complex_line, simple_line in yield_lines_in_parallel([complex_filepath, simple_filepath], strict=False):
                output_files.write(self.encode_sentence_pair(complex_line, simple_line))

class FeaturePreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(
        self,
        feature_name,
        get_feature_value,
        get_target_feature_value,
        bucket_size=0.05,
        noise_std=0,
        prepend_to_target=False,
        use_short_name=False,
    ):
        self.get_feature_value = get_feature_value
        self.get_target_feature_value = get_target_feature_value
        self.bucket_size = bucket_size
        self.noise_std = noise_std
        self.feature_name = feature_name.upper()
        self.use_short_name = use_short_name
        if use_short_name:
            self.feature_name = self.feature_name.lower()[:4]
        self.prepend_to_target = prepend_to_target

    def get_hash_string(self):
        return f'{self.__class__.__name__}(feature_name={repr(self.feature_name)}, bucket_size={self.bucket_size}, noise_std={self.noise_std}, prepend_to_target={self.prepend_to_target}, use_short_name={self.use_short_name})'

    def bucketize(self, value):
        return round(round(value / self.bucket_size) * self.bucket_size, 10)

    def add_noise(self, value):
        return value + np.random.normal(0, self.noise_std)

    def get_feature_token(self, feature_value):
        return f'<{self.feature_name}_{feature_value}>'

    def encode_sentence(self, sentence, encoder_sentence=None):
        if not self.prepend_to_target:
            desired_feature = self.bucketize(self.get_target_feature_value(remove_special_tokens(sentence)))
            sentence = f'{self.get_feature_token(desired_feature)} {sentence}'
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        if self.prepend_to_target:
            _, sentence = extract_special_tokens(sentence)
        return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        feature = self.bucketize(
            self.add_noise(
                self.get_feature_value(remove_special_tokens(complex_sentence), remove_special_tokens(simple_sentence))
            )
        )
        if self.prepend_to_target:
            simple_sentence = f'{self.get_feature_token(feature)} {simple_sentence}'
        else:
            complex_sentence = f'{self.get_feature_token(feature)} {complex_sentence}'
        return complex_sentence, simple_sentence

class LevenshteinPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(), self.get_feature_value, self.get_target_feature_value, bucket_size, noise_std, **kwargs
        )

    @staticmethod
    def get_nevergrad_variables():
        return  ng.p.TransitionChoice([0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])

    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_levenshtein_similarity(complex_sentence, simple_sentence)

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio

class ReplaceOnlyLevenshteinPreprocessor(LevenshteinPreprocessor):
    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_replace_only_levenshtein_similarity(complex_sentence, simple_sentence)

class RatioPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, feature_extractor, target_ratio=0.8, bucket_size=0.05, noise_std=0, **kwargs):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio
        super().__init__(
            self.prefix.upper(), self.get_feature_value, self.get_target_feature_value, bucket_size, noise_std, **kwargs
        )

    @staticmethod
    def get_nevergrad_variables():
        return  ng.p.TransitionChoice([0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5])

    def get_feature_value(self, complex_sentence, simple_sentence):
        return min(
            failsafe_division(self.feature_extractor(simple_sentence), self.feature_extractor(complex_sentence)), 2
        )

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio

class LengthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(len, *args, **kwargs)

class WordRankRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language='en', **kwargs):
        super().__init__(lambda sentence: get_lexical_complexity_score(sentence, language=language), *args, **kwargs)

class DependencyTreeDepthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, language='en', **kwargs):
        super().__init__(lambda sentence: get_dependency_tree_depth(sentence, language=language), *args, **kwargs)