from generate_components_hmm import main
from viterbi import viterbi
import numpy as np
import nltk


# From other python script, obtain transition matrix, initial state probabilities, and tag/word indices.
transition, initial_state, observation, tag_indices, word_indices = main()

# Import test corpus
test_corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]

# Testing function which takes all of the output from Q1, and the test corpus.
def testing(
    transition, initial_state, observation, tag_indices, word_indices, test_corpus
):
    index_tag = {}
    index_word = {}
    for key, val in tag_indices.items():
        index_tag[val] = key
        pass

    for key, val in word_indices.items():
        index_word[val] = key
        pass

    missed = []
    testing = []
    tracking = []
    performance = {"correct": 0, "incorrect": 0}
    for i in range(len(test_corpus)):
        sentence = test_corpus[i]
        correct = []
        obs = []
        for word in sentence:
            correct.append(tag_indices[word[1]])
            if word[0].lower() in word_indices:
                obs.append(word_indices[word[0].lower()])
                pass
            else:
                obs.append(word_indices["<unk>"])
                pass
            pass
        output = viterbi(obs, initial_state, transition, observation)
        for j in range(len(output)):
            if output[j] == correct[j]:
                performance["correct"] += 1
                pass
            else:
                performance["incorrect"] += 1
                original_word = sentence[j][0]
                original_pos = sentence[j][1]
                predic_pos = index_tag[output[j]]
                missed.append(obs[j])
                if obs[j] == word_indices["<unk>"]:
                    unk = 1
                    pass
                else:
                    unk = 0
                    pass
                tracking.append((original_word, original_pos, predic_pos, unk))
                pass
            pass
        pass
    unk = word_indices["<unk>"]
    unknown = sum(x == unk for x in missed)
    percent = round(
        performance["correct"]
        / (performance["correct"] + performance["incorrect"])
        * 100,
        2,
    )
    unk_inc = round(unknown * 100 / performance["incorrect"], 2)
    print(
        f"In this test run, the algorithm identifies words {percent}% correctly. Of all incorrect words, {unk_inc}% of them were unknown words."
    )
    for tuple in tracking:
        statement = f"Our code incorrectly identified '{tuple[0]}' as a {tuple[2]} when it is a {tuple[1]} in the given context."
        if tuple[3] == 1:
            statement += f" This word was unknown to our model."
            pass
        print(statement)
        pass
    pass


if __name__ == "__main__":
    testing(
        transition, initial_state, observation, tag_indices, word_indices, test_corpus
    )
