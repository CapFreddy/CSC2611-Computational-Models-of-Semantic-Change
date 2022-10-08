import numpy as np
import matplotlib.pyplot as plt


def align_historical_embeddings(embeddings):
    # Align to the last snapshot by orthogonal Procrustes.
    tTx = np.matmul(embeddings[-1].T, embeddings)
    u, _, vh = np.linalg.svd(tTx)
    w = np.matmul(vh.transpose(0, 2, 1), u.transpose(0, 2, 1))
    aligned = np.matmul(embeddings, w)
    return aligned


def build_displacement_series(embeddings, word_index=None):
    if word_index is not None:
        embeddings = embeddings[:, word_index, :]
        cos_sim = np.einsum('j,ij->i', embeddings[0], embeddings) \
                  / np.linalg.norm(embeddings[0]) \
                  / np.linalg.norm(embeddings, axis=1)
    else:
        cos_sim = np.einsum('jk,ijk->ij', embeddings[0], embeddings) \
                  / np.linalg.norm(embeddings[0], axis=1) \
                  / np.linalg.norm(embeddings, axis=2)

    dists = 1 - cos_sim
    return dists


def end_displacement_to_start(disp_all):
    return disp_all[-1]


def max_displacement_to_start(disp_all):
    return np.max(disp_all[1:], axis=0)  # Exclude start to start.


def mean_displacement_to_start(disp_all):
    return np.mean(disp_all[1:], axis=0)  # Exclude start to start.


def most_least_changed_k(score_all, words, k):
    sorted_args = np.argsort(score_all)
    most_k = sorted_args[-k :]
    least_k = sorted_args[: k]

    # `most_k`: [..., nan, ..., nan].
    start = -k
    end = len(sorted_args)
    while np.isnan(score_all[most_k]).any():
        assert not np.isinf(score_all[most_k]).any()
        num_replace = np.isnan(score_all[most_k]).sum()
        start -= num_replace
        end -= num_replace
        most_k = sorted_args[start : end]

    # `least_k`: [-inf, ..., -inf, ...]
    start, end = 0, k
    while np.isinf(score_all[least_k]).any():
        assert not np.isnan(score_all[least_k]).any()
        num_replace = np.isinf(score_all[least_k]).sum()
        start += num_replace
        end += num_replace
        least_k = sorted_args[start : end]

    most_k = [words[i] for i in most_k[::-1]]
    least_k = [words[i] for i in least_k]
    return most_k, least_k


def change_point_detection(disp_word, times):
    max_t, max_shift = 0, 0
    for t in range(1, len(disp_word) - 1):
        mean_shift = np.abs(np.mean(disp_word[: t]) - np.mean(disp_word[t :]))
        if mean_shift > max_shift:
            max_shift = mean_shift
            max_t = t

    return times[max_t]


def plot_time_course(disp_word, times, output_path):
    plt.ylabel('Distance')
    plt.xticks(times)
    plt.plot(times, disp_word)
    plt.scatter([times[1]], [disp_word[1]], s=75, color='red', marker='x')
    plt.savefig(output_path, dpi=300)
    plt.clf()


def cloest_k(embeddings, word, words, k):
    cos_sim = np.einsum('j,ij->i', embeddings[words.index(word)], embeddings) \
              / np.linalg.norm(embeddings[words.index(word)]) \
              / np.linalg.norm(embeddings, axis=1)
    sorted_args = np.argsort(cos_sim)
    top_k = sorted_args[-k :]

    # `top_k`: [..., nan, ..., nan].
    start, end = -k, len(sorted_args)
    while np.isnan(cos_sim[top_k]).any():
        assert not np.isinf(cos_sim[top_k]).any()
        num_replace = np.isnan(cos_sim[top_k]).sum()
        start -= num_replace
        end -= num_replace
        top_k = sorted_args[start : end]

    top_k = [words[i] for i in top_k[::-1]]
    return top_k
