# Homework 4 for Sydney Donati-Leach and Raza Lamb
import nltk
import numpy as np


def main():
    # generate training set
    training = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]

    # create unique list of tags
    tags = []
    words = []
    for i in training:
        for j in i:
            tags.append(j[1])
            words.append(j[0].lower())
            pass
    words.append("<unk>")
    tags = set(tags)
    words = set(words)
    words_size = len(words)
    tag_size = len(tags)
    tag_indices = {char: idx for idx, char in enumerate(tags)}
    word_indices = {char: idx for idx, char in enumerate(words)}
    initial_state = np.ones((1, tag_size))

    # make transition matrix and initial state distribution with smoothing
    transition = np.ones((tag_size, tag_size))
    for sentence in training:
        first_state = sentence[0][1]
        add = tag_indices[first_state]
        initial_state[0, add] += 1
        for index in range(1, len(sentence)):
            current_tag = sentence[index][1]
            previous_tag = sentence[index - 1][1]
            transition[tag_indices[previous_tag], tag_indices[current_tag]] += 1
            pass
        pass

    # make observation matrix
    observation = np.ones((tag_size, words_size))
    for sentence in training:
        for word in sentence:
            row = tag_indices[word[1]]
            column = word_indices[word[0].lower()]
            observation[row, column] += 1
            pass
        pass
    # Fix unknown tagging
    # observation[:, word_indices["<unk>"]] = np.sum(observation, axis=1)
    # change to probabilities, then log probatilieies
    sums_t = transition.sum(axis=1)
    transition = transition / sums_t[:, np.newaxis]
    initial_state = initial_state / initial_state.sum()
    sums_o = observation.sum(axis=1)
    observation = observation / sums_o[:, np.newaxis]
    transition = np.log(transition)
    initial_state = np.log(initial_state)
    observation = np.log(observation)
    return transition, initial_state, observation, tag_indices, word_indices
