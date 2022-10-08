import nltk
from gensim.models import KeyedVectors

import numpy as np
import scipy.sparse
import sklearn.decomposition


def build_word_context_model(corpus, vocab):
    # Construct word-context matrix by collecting bigram counts.
    data, row_ind, col_ind = [], [], []

    bigram_freq = nltk.FreqDist(nltk.ngrams(corpus, 2))
    for bigram, freq in bigram_freq.items():
        if bigram[0] in vocab and bigram[1] in vocab:
            data.append(freq)
            row_ind.append(vocab[bigram[0]])
            col_ind.append(vocab[bigram[1]])

    word_context = scipy.sparse.csr_array(
        (data, (row_ind, col_ind)), shape=(len(vocab), len(vocab))
    )
    return word_context, _matrix_to_keyedvector(word_context, vocab)


def build_PPMI_model(word_context, vocab):
    # Compute PPMI based on word-context matrix.
    total = word_context.sum()
    word_cnt = word_context.sum(axis=1)
    context_cnt = word_context.sum(axis=0)

    joint = word_context / total
    marginal_word = word_cnt / total
    marginal_context = context_cnt / total

    # Avoid dividing by zero, will not affect the result.
    marginal_word[marginal_word == 0.] = 1.
    marginal_context[marginal_context == 0.] = 1.

    mi = joint \
         * np.expand_dims(np.reciprocal(marginal_word), axis=0).T \
         * np.reciprocal(marginal_context)

    pmi_data = np.log2(mi.data)
    ppmi_data = np.maximum(pmi_data, 0.)
    ppmi = scipy.sparse.csr_array((ppmi_data, (mi.row, mi.col)), shape=mi.shape)
    return ppmi, _matrix_to_keyedvector(ppmi, vocab)


def build_LSA_model(word_context, vocab, n_components):
    # Construct latent semantic model by applying PCA.
    if scipy.sparse.isspmatrix(word_context):
        word_context = word_context.todense()

    pca = sklearn.decomposition.PCA(n_components, random_state=42)
    lsa = pca.fit_transform(word_context)
    return lsa, _matrix_to_keyedvector(lsa, vocab)


def build_word2vec_model(vocab):
    # Load word2vec embeddings, filter by `vocab`.
    model = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin',
                                              binary=True)
    word2vec_model = KeyedVectors(model.vector_size)
    keys = [word for word in vocab if word in model]
    weights = [model[word] for word in keys]
    word2vec_model.add_vectors(keys, weights)
    return word2vec_model


def _matrix_to_keyedvector(matrix, vocab):
    if scipy.sparse.isspmatrix(matrix):
        matrix = matrix.todense()

    model = KeyedVectors(matrix.shape[1])
    model.add_vectors(list(vocab.keys()), matrix)
    return model
