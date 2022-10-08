from collections import OrderedDict

import nltk
import pandas as pd
from prettytable import PrettyTable

from embedding import (
    build_word_context_model,
    build_PPMI_model,
    build_LSA_model,
    build_word2vec_model
)
from testing import word_similarity_test, word_analogy_test


def main():
    corpus, vocab = build_vocab()

    word_context, word_context_model = build_word_context_model(corpus, vocab)
    ppmi, ppmi_model = build_PPMI_model(word_context, vocab)
    _, lsa_10_model = build_LSA_model(ppmi, vocab, n_components=10)
    _, lsa_100_model = build_LSA_model(ppmi, vocab, n_components=100)
    _, lsa_300_model = build_LSA_model(ppmi, vocab, n_components=300)
    word2vec_model = build_word2vec_model(vocab)

    table_sim = PrettyTable()
    table_ana = PrettyTable()

    table_sim.field_names = ['Model', 'Pearson R (p-value)']
    table_ana.field_names = ['Model', 'Semantic Acc (%)', 'Syntactic Acc (%)']

    model_names = ['word_context', 'ppmi', 'lsa10', 'lsa100', 'lsa300', 'word2vec']
    models = [word_context_model, ppmi_model, lsa_10_model, lsa_100_model, lsa_300_model, word2vec_model]
    for model_name, model in zip(model_names, models):
        stat, pvalue = word_similarity_test(model)
        sem_acc, syn_acc = word_analogy_test(model)

        table_sim.add_row([model_name, '%.2f (%.2f)' % (stat, pvalue)])
        table_ana.add_row([model_name, '%.2f' % (sem_acc*100), '%.2f' % (syn_acc*100)])

    print(table_sim)
    print(table_ana)


def build_vocab():
    # nltk.download('brown')
    corpus = list(nltk.corpus.brown.words())

    # Extract the 5000 most common English words based on unigram frequencies.
    is_eng = lambda word: any(map(str.isalpha, word))
    eng_words = list(filter(is_eng, corpus))

    vocab = OrderedDict()
    unigram_freq = nltk.FreqDist(nltk.ngrams(eng_words, 1))
    for idx, item in enumerate(unigram_freq.most_common(5000)):
        vocab[item[0][0]] = idx

    # Update `vocab` by words in Table 1 of RG65.
    df = pd.read_csv('./data/table1-rg65.csv', header=None)
    for word in pd.concat([df[0], df[1]]):
        if word not in vocab:
            vocab[word] = len(vocab)

    return corpus, vocab


if __name__ == '__main__':
    main()
