from typing import List
import numpy as np


def viterbi_decode(h, mask, start_trans, trans_matrix, end_trans) -> List[List[int]]:
    """
    decode labels using viterbi algorithm
    :param h: hidden matrix (batch_size, seq_len, num_labels)
    :param mask: mask tensor of each sequence
                 in mini batch (batch_size, batch_size)
    :return: labels of each sequence in mini batch
    """
    batch_size = h.shape[0]
    seq_len = h.shape[1]
    seq_lens = np.sum(mask, axis=1)
    print(start_trans)
    print(11111111111111111111111111111)
    print(len(start_trans))
    print(len(h[:, 0]))
    print(h[:, 0])
    score = [start_trans + h[:, 0]]
    path = []

    for t in range(1, seq_len):
        # view(batch_size, -1, 1)
        previous_score = np.expand_dims(score[t - 1], axis=-1)
        h_t = np.expand_dims(h[:, t], axis=1)
        score_t = previous_score + trans_matrix + h_t
        # 找到每行的最大值索引
        # 找到每行的最大值
        best_score = np.max(score_t, axis=1)
        # 找到每行最大值的索引
        best_path = np.argmax(score_t, axis=1)
        score.append(best_score)
        path.append(best_path)
    best_paths = [
        _viterbi_compute_best_path(i, seq_lens, score, path, end_trans)
        for i in range(batch_size)
    ]
    return best_paths


def _viterbi_compute_best_path(
        batch_idx: int,
        seq_lens,
        score,
        path,
        end_trans
) -> List[int]:
    """
    return labels using viterbi algorithm
    :param batch_idx: index of batch
    :param seq_lens: sequence lengths in mini batch (batch_size)
    :param score: transition scores of length max sequence size
                  in mini batch [(batch_size, num_labels)]
    :param path: transition paths of length max sequence size
                 in mini batch [(batch_size, num_labels)]
    :return: labels of batch_idx-th sequence
    """

    seq_end_idx = seq_lens[batch_idx] - 1
    # 函数找到每行最大值的索引
    best_last_label = np.argmax((score[seq_end_idx][batch_idx] + end_trans), axis=0)
    best_labels = [int(best_last_label)]
    for p in reversed(path[:seq_end_idx]):
        best_last_label = p[batch_idx][best_labels[0]]
        best_labels.insert(0, int(best_last_label))

    return best_labels
