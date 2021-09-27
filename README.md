# Part-of-Speech Tagger

This repository contains the Python scripts necessary to implement a POS tagger using a Hidden Markov Model. In the test corpus, the model performs with >90% accuracy.

There are three Python scripts. One builds the main components of the HMM: the transition state matrix, the observation matrix, and the initial state probabilities. The second file contains the program for the Viterbi Algorithm itself, and the last file contains testing script, using the Brown corpus from the NLTK package.

This model used plus-one smoothing everywhere and includes an out-of-vocabulary observation.
