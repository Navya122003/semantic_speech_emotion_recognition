import tensorflow as tf
from tensorflow.keras import layers, Model

def recurrent_model(inputs, emb=None, hidden_units=256, number_of_outputs=3):
    fused_features = attention_model(inputs, emb, projected_units=1024)

    net_1 = tf.keras.layers.Dense(512, activation='relu')(fused_features)
    net_2 = tf.keras.layers.Dense(512, activation='relu')(fused_features)
    net_3 = tf.keras.layers.Dense(512, activation='relu')(fused_features)

    attn_1 = attention_model(net_1, net_2, scope='_self12')
    net = attention_model(attn_1, net_3, scope='_self123')

    lstm_layer = tf.keras.layers.LSTM(hidden_units, return_sequences=True)
    outputs = lstm_layer(net)

    prediction = tf.keras.layers.Dense(number_of_outputs, activation='tanh')(outputs)

    return prediction

class AudioModel2(Model):
    def __init__(self):
        super(AudioModel2, self).__init__()
        self.conv1 = layers.Conv1D(50, 8, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling1D(pool_size=10, strides=10)
        self.drop1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv1D(125, 6, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling1D(pool_size=5, strides=5)
        self.drop2 = layers.Dropout(0.5)

        self.conv3 = layers.Conv1D(250, 6, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling1D(pool_size=5, strides=5)
        self.drop3 = layers.Dropout(0.5)

    def call(self, audio_frames):
        batch_size, seq_length, num_features = tf.shape(audio_frames)[0], tf.shape(audio_frames)[1], tf.shape(audio_frames)[2]
        
        audio_input = tf.reshape(audio_frames, [batch_size, num_features * seq_length, 1])

        net = self.conv1(audio_input)
        net = self.pool1(net)
        net = self.drop1(net)

        net = self.conv2(net)
        net = self.pool2(net)
        net = self.drop2(net)

        net = self.conv3(net)
        net = self.pool3(net)
        net = self.drop3(net)

        net = tf.reshape(net, [batch_size, seq_length, -1])

        return net


def fully_connected_model(audio_frames, text_frames):
    audio_features = layers.Dense(1024, activation='relu')(audio_frames)
    text_features = layers.Dense(1024, activation='relu')(text_frames)

    net = tf.concat([audio_features, text_features], axis=2)
    net = layers.Dense(512, activation='relu')(net)

    return net


def attention_model(audio_frames, text_frames, projected_units=2048, scope=''):
    batch_size, seq_length, num_features = tf.shape(audio_frames)[0], tf.shape(audio_frames)[1], tf.shape(audio_frames)[2]
    
    audio_features = tf.reshape(audio_frames, [-1, num_features])
    text_features = tf.reshape(text_frames, [-1, tf.shape(text_frames)[-1]])
    

    if audio_features.shape[1] != text_features.shape[1]:
        projected_audio = layers.Dense(projected_units, activation='relu')(audio_features)
        projected_text = layers.Dense(projected_units, activation='relu')(text_features)
    else:
        projected_audio = audio_features
        projected_text = text_features
    
    net = stack_attention_producer(projected_audio, projected_text, batch_size, 'attn')

    return net


def stack_attention_producer(frame1, frame2, batch_size, scope=None):
    with tf.name_scope(scope or 'attention_scope'):
        frame1 = tf.expand_dims(frame1, 1)  
        frame2 = tf.expand_dims(frame2, 1)  

        frames = tf.concat([frame1, frame2], axis=1) 
        tmp_frames = tf.expand_dims(frames, axis=3)  
        print(frame1.shape, frame2.shape, frames.shape, tmp_frames.shape)

        conv = tf.keras.layers.Conv2D(
            filters=1,  
            kernel_size=(1, 1),  
            activation=None,
            name=scope + '_selector' if scope else 'selector'
        )(tmp_frames)  

        print(conv.shape)

        conv = tf.reduce_sum(conv, axis=[2, 3])  
        conv = tf.multiply(conv, 1. / tf.sqrt(tf.cast(tf.shape(frames)[-1], tf.float32)))

        attention = tf.nn.softmax(conv, axis=1, name=scope + '_softmax' if scope else 'softmax')
        attention = tf.expand_dims(attention, axis=2)  

        out = tf.reduce_sum(tf.multiply(frames, attention), axis=1, keepdims=True)  

        out = tf.reshape(out, (batch_size, -1, out.shape[-1]))  
        return out



def conv2d(input_, output_dim, k_h, k_w, padding='VALID', name='conv2d'):
    return layers.Conv2D(output_dim, (k_h, k_w), padding=padding)(input_)


def get_model(name):
    name_to_fun = {'audio_model2': audio_model2}

    if name in name_to_fun:
        model_function = name_to_fun[name]  
    else:
        raise ValueError(f'Requested name [{name}] not a valid model')

    def wrapper(*args, **kwargs):
        output = recurrent_model(model_function(*args), **kwargs)
        return output  

    return wrapper

