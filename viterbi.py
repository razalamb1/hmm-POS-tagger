import nltk
import numpy as np

# Viterbi algorithm
def viterbi(obs, pi, A, B):
    num_states = A.shape[0]
    num_words = len(obs)
    vit = np.zeros((num_states, num_words))
    backpointer = np.zeros((num_states, num_words))
    vit[:, 0] = pi + B[:, obs[0]]
    for column in range(1, num_words):
        for row in range(num_states):
            word = obs[column]
            vit[row, column] = np.max(vit[:, column - 1] + A[:, row] + B[row, word])
            backpointer[row, column] = np.argmax(
                vit[:, column - 1] + A[:, row] + B[row, word]
            )
            pass
        pass
    types = np.empty(num_words, "B")
    types[-1] = np.argmax(vit[:, num_words - 1])
    for i in reversed(range(1, num_words)):
        types[i - 1] = backpointer[types[i], i]
        pass
    return list(types)
