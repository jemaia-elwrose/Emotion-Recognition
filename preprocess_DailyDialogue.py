import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def encode_labels(encoder, l):
    """
    Helper function to encode the labels.

    """

    return encoder[l]


def preprocess_text(x):
    """
    Function to preprocess the given sentence by removing punctuations.

    Args:
        x: sentence to be preprocessed

    Returns:
        x: processed sentence
        
    """

    for punct in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
        x = x.replace(punct, ' ')

    x = ' '.join(x.split())
    x = x.lower()

    return x


def create_utterances(filename, split):
    """
    Function to extract information from json file.

    Args:
        filename: jason filename
        split: train/valid/test

    Returns:
        data: pandas dataframe contains all the info
        
    """

    sentences, act_labels, emotion_labels = [], [], []
    speakers, conv_id, utt_id = [], [], []

    lengths = []
    with open(filename, 'r') as f:
        for c_id, line in enumerate(f):
            s = eval(line)

            for u_id, items in enumerate(s['dialogue']):
                sentences.append(items['text'])
                act_labels.append(items['act'])
                emotion_labels.append(items['emotion'])
                conv_id.append(split[:2] + '_c' + str(c_id))
                utt_id.append(split[:2] + '_c' + str(c_id) + '_u' + str(u_id))
                speakers.append(str(u_id % 2))


    data = pd.DataFrame(data=sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))
    data['act_label'] = act_labels
    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id
    data['utt_id'] = utt_id

    return data


def prepare_data(path, split, tokenizer=None):
    """
    Helper function to prepare data like tokenizing, padding, label encoding.

    Args:
        path: path to data.json file
        split: train/valid/test
        tokenizer: ketas pre-fitted tokenizer

    Returns:
        data_df: pandas dataframe contains data info
        emotion_label_decoder: dictionary -> id: label
        
    """

    data_df = create_utterances(path, split)

    # Encoding the emotion and dialog act labels
    all_act_labels = set(data_df['act_label'])
    all_emotion_labels = set(data_df['emotion_label'])
    
    act_label_encoder, emotion_label_encoder = {}, {}
    act_label_decoder, emotion_label_decoder = {}, {}

    for i, label in enumerate(all_act_labels):
        act_label_encoder[label] = i
        act_label_decoder[i] = label

    for i, label in enumerate(all_emotion_labels):
        emotion_label_encoder[label] = i
        # print(f' {i} -> {label}') # For visualization
        emotion_label_decoder[i] = label
        print(f'{emotion_label_encoder[label]} -> {emotion_label_decoder[i]}')

    data_df['encoded_act_label'] = data_df['act_label'].map(
        lambda x: encode_labels(act_label_encoder, x))
    data_df['encoded_emotion_label'] = data_df['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))

    # Tokenizing all sentences using keras
    if tokenizer == None:
        all_text = list(data_df['sentence'])
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_text)

    # Converting the sentences into sequences
    sequence = tokenizer.texts_to_sequences(list(data_df['sentence']))
    data_df['sentence_length'] = [len(item) for item in sequence]

    # Padding sequences
    max_num_tokens = 250
    sequence = pad_sequences(sequence, maxlen=max_num_tokens, padding='post')
    data_df['sequence'] = list(sequence)

    return data_df, emotion_label_decoder


def load_pretrained_glove(path):
    """
    Function to load glove pretrained weights.

    Args:
        path: path to glove file

    Returns:
        glv_vector: loaded pretrain weights
        
    """

    print(f'Loading GloVe model...')
    glv_vector = {}

    f = open(path, encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]

        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs

        except ValueError:
            continue

    f.close()
    print('Completed loading pretrained GloVe model.')

    return glv_vector



if __name__ == '__main__':

    train_data = create_utterances('dailydialog/train.json', 'train')
    valid_data = create_utterances('dailydialog/dev.json', 'valid')
    test_data = create_utterances('dailydialog/test.json', 'test')

    # Encoding the emotion and dialog act labels
    all_act_labels = set(train_data['act_label'])
    all_emotion_labels = set(train_data['emotion_label'])
    act_label_encoder, emotion_label_encoder = {}, {}
    act_label_decoder, emotion_label_decoder = {}, {}

    for i, label in enumerate(all_act_labels):
        act_label_encoder[label] = i
        act_label_decoder[i] = label

    for i, label in enumerate(all_emotion_labels):
        emotion_label_encoder[label] = i
        print(f'{i} -> {label}')

        emotion_label_decoder[i] = label
        print(f'{emotion_label_encoder[label]} -> {emotion_label_decoder[i]}')

##    pickle.dump(act_label_encoder, open('dailydialog/act_label_encoder.pkl', 'wb'))
##    pickle.dump(act_label_decoder, open('dailydialog/act_label_decoder.pkl', 'wb'))
##    pickle.dump(emotion_label_encoder, open('dailydialog/emotion_label_encoder.pkl', 'wb'))
##    pickle.dump(emotion_label_decoder, open('dailydialog/emotion_label_decoder.pkl', 'wb'))

    train_data['encoded_act_label'] = train_data['act_label'].map(
        lambda x: encode_labels(act_label_encoder, x))
    test_data['encoded_act_label'] = test_data['act_label'].map(
        lambda x: encode_labels(act_label_encoder, x))
    valid_data['encoded_act_label'] = valid_data['act_label'].map(
        lambda x: encode_labels(act_label_encoder, x))

    train_data['encoded_emotion_label'] = train_data['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))
    test_data['encoded_emotion_label'] = test_data['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))
    valid_data['encoded_emotion_label'] = valid_data['emotion_label'].map(
        lambda x: encode_labels(emotion_label_encoder, x))

    # Tokenizing all sentences using keras
    all_text = list(train_data['sequence'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)
    pickle.dump(tokenizer, open('dailydialog/tokenizer.pkl', 'wb'))

    # Converting the sentences into sequences
    train_sequence = tokenizer.texts_to_sequences(list(train_data['sentence']))
    valid_sequence = tokenizer.texts_to_sequences(list(valid_data['sentence']))
    test_sequence = tokenizer.texts_to_sequences(list(test_data['sentence']))

    train_data['sentence_length'] = [len(item) for item in train_sequence]
    valid_data['sentence_length'] = [len(item) for item in valid_sequence]
    test_data['sentence_length'] = [len(item) for item in test_sequence]

    max_num_tokens = 250

    train_sequence = pad_sequence(train_sequence, maxlen=num_tokens, padding='post')
    valid_sequence = pad_sequence(valid_sequence, maxlen=num_tokens, padding='post')
    test_sequence = pad_sequence(test_sequence, maxlen=num_tokens, padding='post')

    train_data['sequence'] = list(train_sequence)
    valid_data['sequence'] = list(valid_sequence)
    test_data['sequence'] = list(test_sequence)

    # Save the data in pickle format
    convSpeakers, convInputSequence = {}, {}
    convInputMaxSequenceLength = {}
    convActLabels, convEmotionLabels = {}, {}

    train_conv_ids = set(train_data['conv_id'])
    test_conv_ids = set(test_data['conv_id'])
    valid_conv_ids = set(valid_data['conv_id'])
    all_data = train_data\
               .append(test_data, ignore_index=True)\
               .append(valid_data, ignore_index=True)

    print('Preparing Data...')
    for item in list(train_conv_ids) \
        + list(test_conv_ids) \
        + list(valid_conv_ids):
        
        df = all_data[all_data['conv_id'] == item]

        convSpeakers[item] = list(df['speakers'])
        convInputSequence[item] = list(df['sequence'])
        convInputMaxSequenceLength[item] = max(list(df['sentence_length']))
        convActLabels[item] = list(df['encoded_act_label'])
        convEmotionLabels[item] = list(df['encoded_emotion_label'])

    pickle.dump([convSpeakers, convInputSequence, convInputMaxSequence,
                 convActLabels, convEmotionLabels, train_conv_ids,
                 test_conv_ids, valid_conv_ids], open('dailydialog/daily_dialogue.pkl', 'wb'))

    # Save pretrained embedding matrix
    glv_vector = load_pretrained_glove()
    word_vector_length = len(glv_vector['the'])
    word_index = tokenizer.word_index
    inv_word_index = {v: k for k, v in word_index.items()}
    num_unique_words = len(word_index)
    glv_embedding_matrix = np.zeros((num_unique_words + 1, word_vector_length))

    for j in range(1, num_unique_words + 1):
        try:
            glv_embedding_matricx[j] = glv_vector[inv_word_index[j]]
        except KeyError:
            glv_embbeding_matric[j] = np.random.randn(word_vector_length) / 200

    np.ndarray.dump(glv_embbeding_matrix, open('dailydialog/glv_embedding_matrix', 'wb'))
    print('Done Completed Preprocessing...')
                                            

    
