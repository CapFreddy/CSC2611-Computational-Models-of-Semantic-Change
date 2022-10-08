import pickle
import numpy as np
from prettytable import PrettyTable

from semantic_change import (
    align_historical_embeddings,
    build_displacement_series,
    end_displacement_to_start,
    max_displacement_to_start,
    mean_displacement_to_start,
    most_least_changed_k,
    change_point_detection,
    plot_time_course,
    cloest_k
)
from testing import semantic_change_test, correlation_test


def main():
    data = pickle.load(open('./embeddings/data.pkl', 'rb'))
    words, times = data['w'], data['d']
    embeddings = np.array(data['E']).transpose(1, 0, 2)

    embeddings = align_historical_embeddings(embeddings)
    disp_all = build_displacement_series(embeddings)

    table_words = PrettyTable()
    table_corrs = PrettyTable()
    table_change = PrettyTable()

    table_words.field_names = ['Method', 'Most Changing', 'Least Changing']
    table_corrs.field_names = ['', 'end2start', 'max2start', 'mean2start']
    table_change.field_names = ['Method', 'Pearson R (p-value)']

    method_names = ['end2start', 'max2start', 'mean2start']
    method_funcs = [end_displacement_to_start, max_displacement_to_start, mean_displacement_to_start]
    for method_name, method_func in zip(method_names, method_funcs):
        score_all = method_func(disp_all)
        most_20, least_20 = most_least_changed_k(score_all, words, k=20)
        corrs = [correlation_test(score_all, func(disp_all)) for func in method_funcs]
        stat, pvalue = semantic_change_test(score_all, words)

        most_20 = '\n'.join([' '.join([most_20[p] for p in range(k, k+4)]) for k in range(0, 20, 4)])
        least_20 = '\n'.join([' '.join([least_20[p] for p in range(k, k+4)]) for k in range(0, 20, 4)])
        corrs = list(map(lambda corr: '%.2f (%.2f)' % (corr[0], corr[1]), corrs))

        table_words.add_row([method_name, most_20 + '\n', least_20 + '\n'])
        table_corrs.add_row([method_name] + corrs)
        table_change.add_row([method_name, '%.2f (%.2f)' % (stat, pvalue)])

    print(table_words)
    print(table_corrs)
    print(table_change)

    # Best Method: mean2start.
    score_all = mean_displacement_to_start(disp_all)
    most_3, _ = most_least_changed_k(score_all, words, k=3)
    for word in most_3:
        disp_word = disp_all[:, words.index(word)]
        change_point = change_point_detection(disp_word, times)
        plot_time_course(disp_word, times, f'./plot/{word}.png')

        print(change_point)

        for idx, t in enumerate(times):
            print(f'{word}@{t}', cloest_k(embeddings[idx], word, words, 6))


if __name__ == '__main__':
    main()
