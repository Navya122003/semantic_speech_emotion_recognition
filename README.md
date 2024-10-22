Speech Emotion Recognition using Semantic Information
This repository provides training and evaluation code for the paper Speech Emotion Recognition using Semantic Information (ICASSP 2021). If you use this codebase in your experiments please cite:


Steps:
1. Create the speech2vec segmentation by running speech2word_mapping.py.
2. Run generate_tfrecords.py to create tfrecords.
3. Copy the codes helper, models, losses, data_provider, data_utils in the same python script or jupyter notebook for training
4. Run train.py to train the model.
