import os

import librosa as lb
import numpy as np
import tensorflow as tf

from pathlib import Path

# Path to AVEC 2017 SEWA DB
AVEC_DIR = 'C:/Users/gupta/Downloads/small_test_dataset'

# Path to speech2vec segmentation
AUDIO2WORD_TIME_MAPPING_DIR = 'C:/Users/gupta/Downloads/small_test_dataset/segmentation'

SAVE_TFRECORD_DIR = Path('C:/Users/gupta/Downloads/small_test_dataset/tfrecords')  # Please specify your own path to save the tfrecords
SAVE_TFRECORD_DIR.mkdir(exist_ok=True)
(SAVE_TFRECORD_DIR / 'data').mkdir(exist_ok=True)
(SAVE_TFRECORD_DIR / 'words').mkdir(exist_ok=True)
(SAVE_TFRECORD_DIR / 'sentences').mkdir(exist_ok=True)

TARGET_SAMPLING_RATE = 22050
CHUNK_SIZE = 2205  # corresponds to 0.1s frame
MAX_DATA_LENGTH = 50

fnames = ['Train', 'Devel', 'Test']

WORD2VEC_DIR = 'C:/Users/gupta/Downloads/'
W2V_VEC = '300.vec'

SYNONYM_DICT = {
    '<laughter>': ['lachen', 'gelächter'],
    '<slightlaughter>': ['lachen', 'gelächter'],
    '<breathing>': ['atmung'],
    '<moaning>': ['stöhnen', 'seufzen', 'gejammer', 'gestöhne'],
    '<contempt>': ['verachtung', 'geringschätzung'],
    '<fumbling>': ['fummelei', 'gefummel', 'linkisch'],
    '<clearingthroat>': ['räuspern'],
    '<coughing>': ['husten'],
    '<singing>': ['singen', 'gesang'],
    '<disgust>': ['ekeln', 'anwidern', 'empören', 'anekeln'],
    '<clicking>': ['klicken']
}

EMBEDDING_DIMENSION = 300

class WordVectorHelper(object):
    def __init__(self, path):
        self.path = path

    def load_vec(self):
        embeddings = []
        embeddings_dict = {}

        id2word = dict()

        with open(self.path, 'r', encoding='utf-8') as f:
            l1 = f.readline().split()
            vocabulary, embedding_size = int(l1[0]), int(l1[1])

            count = 0
            for line in f:
                tmp = line.split()
                try:
                    word, embed = tmp[0], np.array(tmp[1:], dtype=float)

                    if len(embed) == embedding_size:
                        embeddings.append(embed)
                        embeddings_dict[word] = embed
                        id2word[count] = word
                        count += 1
                except:
                    print('Cant process word {%s}' % tmp[0])

        word2id = dict(zip(id2word.values(), id2word.keys()))
        print('Vocabulary size: %d, embedding size: %d' % (len(embeddings), embedding_size))

        self.embeddings_dict = embeddings_dict
        self.id2word = id2word
        self.word2id = word2id
        self.embeddings = embeddings

        return id2word, word2id, embeddings, embeddings_dict

    def check_for_synonym_in_vec(self):
        # Need to call load_vec() first
        embeddings = self.embeddings_dict

        s_embeddings = dict()
        for k, v in SYNONYM_DICT.items():
            for word in v:
                e = embeddings.get(word, None)
                found = True if e is not None else False
                # print(word, found)

                if found:
                    s_embeddings[k] = e
                    continue

        return s_embeddings

    def get_word_by_embedding(self, vector):
        for i, e in enumerate(self.embeddings):
            if (e == vector).all():
                return self.id2word[i]

        return None

def get_filenames_from_dir(dir):
    trains, vals, tests = [], [], []

    for filename in sorted(os.listdir(dir)):
        fn = filename.split('.')[0]
        if filename.startswith('Train'):
            trains.append(fn)
        elif filename.startswith('Devel'):
            vals.append(fn)
        elif filename.startswith('Test'):
            tests.append(fn)
        else:
            print('Unknown filename: {%s}' % filename)

    return trains, vals, tests

def serialize_sample(writer, filename):
    for _, (audio_frame, embedding, label) in enumerate(zip(*get_samples(filename))):  # serialize every frame
        example = tf.train.Example(features=tf.train.Features(feature={
            'file_name': _bytes_feature(filename.encode()),
            'audio_frame': _bytes_feature(audio_frame.tobytes()),
            'embedding': _bytes_feature(embedding.tobytes()),
            'label': _bytes_feature(label.tobytes()),
        }))

        writer.write(example.SerializeToString())  # write all frames of a subject to a file

def serialize_sentence(writer, filename):
    for i, sent in enumerate(get_sentences(filename)):  # serialize every frame
        audio_frames = sent['audio_frames']

        # print(audio_frames.shape, sent['words'].shape, sent['labels'].shape)

        example = tf.train.Example(features=tf.train.Features(feature={
            'file_name': _bytes_feature(filename.encode()),
            'sentence_id': _int_feature(i),
            'sentence_length': _int_feature(audio_frames.shape[0]),
            'audio_frames': _bytes_feature(audio_frames.tobytes()),
            'words': _bytes_feature(sent['words'].tobytes()),
            'labels': _bytes_feature(sent['labels'].tobytes()),
        }))

        writer.write(example.SerializeToString())  

def serialize_word(writer, filename):
    for i, word in enumerate(get_words(filename)):  # serialize every frame
        audio_frames = word['audio_frames']

        # print(audio_frames.shape, word['embeddings'].shape, word['labels'].shape)

        example = tf.train.Example(features=tf.train.Features(feature={
            'file_name': _bytes_feature(filename.encode()),
            'word_id': _int_feature(i),
            'word_length': _int_feature(audio_frames.shape[0]),
            'audio_frames': _bytes_feature(audio_frames.tobytes()),
            'embeddings': _bytes_feature(word['embeddings'].tobytes()),
            'labels': _bytes_feature(word['labels'].tobytes()),
        }))

        writer.write(example.SerializeToString())  

def get_samples(filename):
    audio_signal, sr, labels, turns, time_mappings = load_metadata(filename)

    # Process labels
    time = labels[:, 1].astype(np.float32)
    arousal = np.reshape(labels[:, 2], (-1, 1)).astype(np.float32)
    valence = np.reshape(labels[:, 3], (-1, 1)).astype(np.float32)
    liking = np.reshape(labels[:, 4], (-1, 1)).astype(np.float32)

    labels = np.hstack([arousal, valence, liking]).astype(np.float32)

    # Process audio frames
    target_interculator_audio = process_audio_frames(time, audio_signal, sr, turns)
    print(len(target_interculator_audio), target_interculator_audio[0].shape)

    # Process word mappings
    corresponding_word, _, _ = process_word_mappings(time, time_mappings)
    print(target_interculator_audio[0].shape, corresponding_word[0].shape, labels[0].shape)

    return target_interculator_audio, corresponding_word, labels

def get_sentences(filename):
    global max_sent_length

    audio_signal, sr, labels, turns, time_mappings = load_metadata(filename)
    

    # Process labels
    time = labels[:, 1].astype(np.float32)
    arousal = np.reshape(labels[:, 2], (-1, 1)).astype(np.float32)
    valence = np.reshape(labels[:, 3], (-1, 1)).astype(np.float32)
    liking = np.reshape(labels[:, 4], (-1, 1)).astype(np.float32)

    labels = np.hstack([arousal, valence, liking]).astype(np.float32)

    # Process audio frames
    target_interculator_audio = process_audio_frames(time, audio_signal, sr, turns)
    print(len(target_interculator_audio), target_interculator_audio[0].shape)

    # Process word mappings
    corresponding_word, corresponding_sentence_id, _ = process_word_mappings(time, time_mappings)

    sentences = []
    saf = np.array(target_interculator_audio[0])  
    sw = np.array(corresponding_word[0])  
    sl = np.array(labels[0])  
    sent_num = 1

    print(target_interculator_audio[0].shape, corresponding_word[0].shape, labels[0].shape)

    for i in range(1, len(time)):
        if corresponding_sentence_id[i] == sent_num:
            saf = np.vstack((saf, target_interculator_audio[i]))
            sw = np.vstack((sw, corresponding_word[i]))
            sl = np.vstack((sl, labels[i]))
        else:
            sentences.append({'audio_frames': saf, 'words': sw, 'labels': sl})

            if saf.shape[0] > max_sent_length:
                print('Longest sentence frames: %s' % max_sent_length)
                max_sent_length = saf.shape[0]

            sent_num = corresponding_sentence_id[i]  
            saf = np.array(target_interculator_audio[i])
            sw = np.array(corresponding_word[i])
            sl = np.array(labels[i])

    sentences.append({'audio_frames': saf, 'words': sw, 'labels': sl})

    return sentences


def get_words(filename):
    global max_word_length

    audio_signal, sr, labels, turns, time_mappings = load_metadata(filename)

    # Process labels
    time = labels[:, 1].astype(np.float32)
    arousal = np.reshape(labels[:, 2], (-1, 1)).astype(np.float32)
    valence = np.reshape(labels[:, 3], (-1, 1)).astype(np.float32)
    liking = np.reshape(labels[:, 4], (-1, 1)).astype(np.float32)

    labels = np.hstack([arousal, valence, liking]).astype(np.float32)

    # Process audio frames
    target_interculator_audio = process_audio_frames(time, audio_signal, sr, turns)
    print(len(target_interculator_audio), target_interculator_audio[0].shape)

    # Process word mappings
    corresponding_word, _, corresponding_word_id = process_word_mappings(time, time_mappings)

    words = []
    af = np.array(target_interculator_audio[0])  
    e = np.array(corresponding_word[0])  
    l = np.array(labels[0])  
    word_num = corresponding_word_id[0]

    print(target_interculator_audio[0].shape, corresponding_word[0].shape, labels[0].shape)

    for i in range(1, len(time)):
        if corresponding_word_id[i] == word_num:
            af = np.vstack((af, target_interculator_audio[i]))
            e = np.vstack((e, corresponding_word[i]))
            l = np.vstack((l, labels[i]))
        else:
            words.append({'audio_frames': af, 'embeddings': e, 'labels': l})

            if af.shape[0] > max_word_length:
                max_word_length = af.shape[0]
                print('Longest word frames: %s' % max_word_length)

        
            word_num = corresponding_word_id[i]  
            af = np.array(target_interculator_audio[i])
            e = np.array(corresponding_word[i])
            l = np.array(labels[i])

    words.append({'audio_frames': af, 'embeddings': e, 'labels': l})

    return words


def load_metadata(filename):
    label_path = AVEC_DIR + '/labels/{}.csv'.format(filename)
    turn_path = AVEC_DIR + '/turns/{}.csv'.format(filename)
    a2w_mapping_path = AUDIO2WORD_TIME_MAPPING_DIR + '/{}.csv'.format(filename)

    audio_signal, sampling_rate = lb.core.load(AVEC_DIR + '/audio/{}.wav'.format(filename), sr=TARGET_SAMPLING_RATE)
    audio_signal = np.pad(audio_signal, (0, CHUNK_SIZE - audio_signal.shape[0] % CHUNK_SIZE), 'constant')

    labels = np.loadtxt(str(label_path), delimiter=',', dtype=str, skiprows=1)
  
    turns = []
    with open(turn_path, 'r') as f:
        next(f)  
        for line in f:
            row = line.strip().split(',')
            if len(row) < 3 or '' in row: 
                print(f"Skipping invalid row in turns file: {row}")
                continue
            try:
                start_time = float(row[1])
                end_time = float(row[2])
                turns.append([start_time, end_time])
            except ValueError:
                print(f"Skipping invalid row due to conversion error: {row}")
    turns = np.array(turns, dtype=np.float32) 
    time_mappings = np.loadtxt(str(a2w_mapping_path), delimiter=';', dtype=str)

    return audio_signal, sampling_rate, labels, turns, time_mappings

def process_audio_frames(time, audio_signal, sr, turns):
    target_interculator_audio = [np.zeros((1, 4410), dtype=np.float32)
                                 for _ in range(len(time))]  

    target_set = set()
    audio_frames = []

    for _, t in enumerate(time): 
        s = int(t * sr)
        e = s + 2205
        audio = np.reshape(audio_signal[s:e], (1, -1))
        audio_frames.append(audio.astype(np.float32))

    for turn in turns:
        st, end = int(round(float(turn[0]), 1) * 10), int(round(float(turn[1]), 1) * 10)
        for i in range(st, end + 1 if end + 1 < len(time) else len(time)):
            target_set.add(i)
            target_interculator_audio[i][0][:2205] = audio_frames[i]  # the subject is speaking

    for i in range(len(time)):
        if i not in target_set:
            target_interculator_audio[i][0][2205:] = audio_frames[i]  # the chatting partner is speaking

    return target_interculator_audio


def process_word_mappings(time, time_mappings):
    # Process word mappings
    id = time_mappings[:, 0]
    start_time = time_mappings[:, 1].astype(np.float32)
    end_time = time_mappings[:, 2].astype(np.float32)
    word = time_mappings[:, 3]

    corresponding_word = [None for _ in range(len(time))]
    corresponding_word_id = [None for _ in range(len(time))]
    corresponding_sentence_id = [0 for _ in range(len(time))]

    for i, w in enumerate(word):
        st, end = int(round(float(start_time[i]), 1) * 10), int(round(float(end_time[i]), 1) * 10)
        for t in range(st, end + 1 if end + 1 < len(time) else len(time)):
            emb = _get_embedding(w)
            corresponding_word[t] = emb.reshape((1, EMBEDDING_DIMENSION))
            corresponding_word_id[t] = id[i]
            corresponding_sentence_id[t] = int(id[i].split('w')[0].split('s')[1])

    sid = -1
    count = 0
    for i in range(len(time)):
        if corresponding_word[i] is None:
            corresponding_word[i] = np.zeros((1, EMBEDDING_DIMENSION))

            if i > 0 and corresponding_word_id[i-1] == sid:
                count = 2 if count == 0 else count + 1
            else:
                count = 0

            corresponding_word_id[i] = sid
            corresponding_sentence_id[i] = sid

            if count == MAX_DATA_LENGTH:
                sid -= 1
                count = 0

    corresponding_word = np.array(corresponding_word)
    return corresponding_word, corresponding_sentence_id, corresponding_word_id


def _get_embeddings(space=''):
    if space == 's2v':
        d, v = SPEECH2VEC_DIR, S2V_VEC
    elif space == 'w2v':
        d, v = WORD2VEC_DIR, W2V_VEC
    else:
        d, v = CROSS_EMBEDDING_DIR, CE_VEC

    vec_helper = WordVectorHelper(d + v)
    id2word, word2id, emb, embed_dic = vec_helper.load_vec()
    syn_embed_dic = vec_helper.check_for_synonym_in_vec()

    return id2word, embed_dic, syn_embed_dic


def _get_embedding(word):
    global unk
    word = _clean_word(word)

    if embed_dict.get(word, None) is not None:
        return embed_dict[word]
    elif syn_dict.get(word, None) is not None:
        return syn_dict[word]
    else:
        unk.add(word)
        embed_dict[word] = np.random.rand(EMBEDDING_DIMENSION)
        print('Word {%s} not in dict, new size of embedding dict {%d}' % (word, len(embed_dict)))
        return embed_dict[word]


def _clean_word(word):
    punctuation = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'  # Exclude <>
    word_clean = word.translate(str.maketrans('', '', punctuation))
    word_clean = word_clean.lower()
    return word_clean


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def write_sentences():
    train, devel, test = get_filenames_from_dir(AVEC_DIR + '/audio')
    fnames_dic = {'Train': train, 'Devel': devel, 'Test': test}

    for fname in fnames:
        for filename in fnames_dic[fname]:
            print('Writing tfrecords for {} file'.format(filename))
            
            writer = tf.io.TFRecordWriter(str(SAVE_TFRECORD_DIR / 'sentences' / (filename + '.tfrecords')))
            serialize_sentence(writer, filename)


def write_words():
    train, devel, test = get_filenames_from_dir(AVEC_DIR + '/audio')
    fnames_dic = {'Train': train, 'Devel': devel, 'Test': test}

    for fname in fnames:
        for filename in fnames_dic[fname]:
            print('Writing tfrecords for {} file'.format(filename))

            writer = tf.io.TFRecordWriter(str(SAVE_TFRECORD_DIR / 'words' / (filename + '.tfrecords')))
            serialize_word(writer, filename)


def write_data():
    train, devel, test = get_filenames_from_dir(AVEC_DIR + '/audio')
    fnames_dic = {'Train': train, 'Devel': devel, 'Test': test}

    for fname in fnames:
        for filename in fnames_dic[fname]:
            print('Writing tfrecords for {} file'.format(filename))

            writer = tf.io.TFRecordWriter(str(SAVE_TFRECORD_DIR / 'data' / (filename + '.tfrecords')))
            serialize_sample(writer, filename)


max_sent_length = 0
max_word_length = 0
unk = set()

if __name__ == '__main__':
    id2word, embed_dict, syn_dict = _get_embeddings(space='w2v')  # space = 'w2v|s2v|<empty>'
    write_sentences()
    write_words()
    write_data()
