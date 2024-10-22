Speech Emotion Recognition using Semantic Information
This repository provides training and evaluation code for the paper Speech Emotion Recognition using Semantic Information (ICASSP 2021). If you use this codebase in your experiments please cite:

Requirements
Below are listed the required modules to run the code.

aeneas
librosa
nltk
numpy
stop-words
tensorflow
torch


Steps:
Create the speech2vec segmentation by running speech2word_mapping.py in speech2vec folder.
Run data_generator.py to create tfrecords.
Run train.py to train the models, and eval.py to evaluate.
