import tensorflow as tf
import argparse

# Update tfrecord directory
TFRECORD_DIR = 'C:/Users/gupta/Downloads/small_test_dataset/tfrecords/data/'
SENT_DATASET_DIR = 'C:/Users/gupta/Downloads/small_test_dataset/tfrecords/sentences/'
WORD_DATASET_DIR = 'C:/Users/gupta/Downloads/small_test_dataset/tfrecords/words/'

slim = tf.keras.layers

FLAGS = {
    'dataset_dir': TFRECORD_DIR,
    'sent_dataset_dir': SENT_DATASET_DIR,
    'word_dataset_dir': WORD_DATASET_DIR,
    'train_dir': 'C:/Users/gupta/Downloads/small_test_dataset/checkpoints/',
    'learning_rate': 0.05,
    'batch_size': 5,
    'hidden_units': 2,
    'model': 'audio_model2',
    'sequence_length': 100,
    'data_unit': None,
    'liking': True
}

def train():
    tf.random.set_seed(1) 
    
    audio_frames, word_embeddings, ground_truth = get_split(
        FLAGS['dataset_dir'], True, FLAGS['batch_size'], seq_length=FLAGS['sequence_length']
    )

    model = AudioModel2()

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS['learning_rate'], beta_1=0.9, beta_2=0.99)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # Forward pass using the model
            prediction = model(audio_frames)

            total_loss = 0
            count = 0

            for i, name in enumerate(['arousal', 'valence', 'liking']):
                count += 1
                pred_single = tf.reshape(prediction[:, :, i], (-1,))
                gt_single = tf.reshape(ground_truth[:, :, i], (-1,))
                
                loss = concordance_cc(pred_single, gt_single)
                mse = tf.reduce_mean(tf.square(pred_single - gt_single))
                
                total_loss += loss / count

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return total_loss

    # Training loop
    for epoch in range(5): 
        loss = train_step()
        print(f'Epoch {epoch + 1}, Loss: {loss}')

if __name__ == '__main__':
    train()
