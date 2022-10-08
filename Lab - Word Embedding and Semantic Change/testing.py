from collections import defaultdict

import numpy as np
import scipy.stats
import pandas as pd
from gensim.models import KeyedVectors


def word_similarity_test(model: KeyedVectors):
    # Examples with OOV words are automatically ignored.
    pearson, _, _ = model.evaluate_word_pairs('./data/table1-rg65.csv',
                                              delimiter=',',
                                              case_insensitive=False)
    return pearson.statistic, pearson.pvalue


def word_analogy_test(model: KeyedVectors):
    # Examples with OOV words are automatically ignored.
    _, sections = model.evaluate_word_analogies('./data/word-test.v1.txt',
                                                case_insensitive=False)
    sem_crrt, sem_total = 0, 0
    syn_crrt, syn_total = 0, 0
    for section in sections:
        if section['section'] == 'Total accuracy':
            continue

        num_crrt = len(section['correct'])
        num_incrrt = len(section['incorrect'])

        if section['section'].startswith('gram'):
            syn_crrt += num_crrt
            syn_total += num_crrt + num_incrrt
        else:
            sem_crrt += num_crrt
            sem_total += num_crrt + num_incrrt

    sem_acc = sem_crrt / sem_total
    syn_acc = syn_crrt / syn_total
    return sem_acc, syn_acc


def correlation_test(score_1: np.ndarray, score_2: np.ndarray):
    is_invalid = np.isnan(score_1) | np.isinf(score_1) | \
                 np.isnan(score_2) | np.isinf(score_2)
    pearson = scipy.stats.pearsonr(score_1[~is_invalid], score_2[~is_invalid])
    return pearson.statistic, pearson.pvalue


def semantic_change_test(score_all: np.ndarray, words: list):
    score_pred = {word: score_all[idx] for idx, word in enumerate(words)}
    score_true = {}

    # Words evaluated by humans.
    word_score = defaultdict(float)
    for i in range(1, 4):
        df = pd.read_csv(f'./data/20WordsEvaluator{i}.csv')
        for _, (word, _, changed, _) in df.iterrows():
            if isinstance(word, str):
                word_score[word] += changed

    score_true.update({word: score/3 for word, score in word_score.items()})

    # Words reported by previous methods.
    df = pd.read_csv(f'./data/ChangedWords.csv')
    score_true.update({word: 1. for word in df.Word})

    x, y = [], []
    for word, score in score_true.items():
        if word in score_pred:
            x.append(score)
            y.append(score_pred[word])

    pearson = scipy.stats.pearsonr(x, y)
    return pearson.statistic, pearson.pvalue
