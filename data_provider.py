import tensorflow as tf

EMBEDDING_DIMENSION = 300

def get_split(dataset_dir, is_training=True, batch_size=32, seq_length=100):
    if is_training:
        train_set, _, _ = get_filenames_from_dir(dataset_dir)

        paths = [dataset_dir + file + '.tfrecords' for file in train_set]

        dataset = tf.data.TFRecordDataset(paths)
        dataset = dataset.shuffle(buffer_size=len(paths))
    else:
        dataset = tf.data.Dataset.list_files(dataset_dir)

    def parse_example(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'label': tf.io.FixedLenFeature([], tf.string),
                'file_name': tf.io.FixedLenFeature([], tf.string),
                'audio_frame': tf.io.FixedLenFeature([], tf.string),
                'embedding': tf.io.FixedLenFeature([], tf.string),
            }
        )
        
        file_name = features['file_name']
        audio_frame = tf.io.decode_raw(features['audio_frame'], tf.float32)
        embedding = tf.io.decode_raw(features['embedding'], tf.float64)
        label = tf.io.decode_raw(features['label'], tf.float32)

        return audio_frame, embedding, label, file_name

    dataset = dataset.map(parse_example)

    dataset = dataset.batch(batch_size)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) 

    for audio_frames, embeddings, labels, file_names in dataset:
        masked_audio_samples = []
        masked_embeddings = []
        masked_labels = []

        for i in range(batch_size):
            file_names_shape = tf.shape(file_names)

            if file_names_shape[0] > i:
                current_file_name = file_names[i]  

                mask = tf.equal(current_file_name, file_names)

                fs = tf.boolean_mask(audio_frames, mask) 
                es = tf.boolean_mask(embeddings, mask)
                ls = tf.boolean_mask(labels, mask)

                fs = tf.cond(tf.shape(fs)[0] < seq_length,
                            lambda: tf.pad(fs, [[0, seq_length - tf.shape(fs)[0]], [0, 0]], "CONSTANT"),
                            lambda: fs)

                es = tf.cond(tf.shape(es)[0] < seq_length,
                            lambda: tf.pad(es, [[0, seq_length - tf.shape(es)[0]], [0, 0]], "CONSTANT"),
                            lambda: es)

                ls = tf.cond(tf.shape(ls)[0] < seq_length,
                            lambda: tf.pad(ls, [[0, seq_length - tf.shape(ls)[0]], [0, 0]], "CONSTANT"),
                            lambda: ls)

                masked_audio_samples.append(fs)
                masked_embeddings.append(es)
                masked_labels.append(ls)
            else:
                print(f"Warning: Index {i} is out of bounds for file_names shape {file_names_shape}")

        masked_audio_samples = tf.stack(masked_audio_samples)
        masked_embeddings = tf.stack(masked_embeddings)
        masked_labels = tf.stack(masked_labels)

        actual_audio_samples_shape = tf.shape(masked_audio_samples)
        
        if actual_audio_samples_shape[0] == batch_size and actual_audio_samples_shape[1] == seq_length and actual_audio_samples_shape[2] == 4410:
            masked_audio_samples = tf.reshape(masked_audio_samples, (batch_size, seq_length, 4410))
        else:
            print(f"Skipping audio samples reshape due to incompatible shape: {actual_audio_samples_shape}")
            continue  

        if tf.shape(masked_embeddings)[0] == batch_size and tf.shape(masked_embeddings)[1] == seq_length and tf.shape(masked_embeddings)[2] == EMBEDDING_DIMENSION:
            masked_embeddings = tf.reshape(masked_embeddings, (batch_size, seq_length, EMBEDDING_DIMENSION))
        else:
            print(f"Skipping embeddings reshape due to incompatible shape: {tf.shape(masked_embeddings)}")
            continue 

        if tf.shape(masked_labels)[0] == batch_size and tf.shape(masked_labels)[1] == seq_length and tf.shape(masked_labels)[2] == 3:
            masked_labels = tf.reshape(masked_labels, (batch_size, seq_length, 3))
        else:
            print(f"Skipping labels reshape due to incompatible shape: {tf.shape(masked_labels)}")
            continue  

        # print("Masked shapes:", masked_audio_samples.shape, masked_embeddings.shape, masked_labels.shape)

    return masked_audio_samples, masked_embeddings, masked_labels
