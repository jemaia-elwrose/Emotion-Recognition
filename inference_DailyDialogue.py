import pickle
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DialogueGCN_DailyModel
from preprocess_DailyDialogue import prepare_data
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    

def tensorize(features, speakers, emotion_labels):
    """
    Helper function to convert given list into tensor format.
    
    """
    textf = torch.FloatTensor(features).unsqueeze(1)
    qmask = torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in speakers]).unsqueeze(1)
    umask = torch.FloatTensor([1]*len(emotion_labels)).unsqueeze(0)
    label = torch.LongTensor(emotion_labels).unsqueeze(0)

    return textf, qmask, umask, label


def make_prediction(data, model):
    """
    Function to make prediction using given data.

    Args:
        data: processed data
        model: model with pretrined weights loaded

    Returns:
        labels, preds, conv_ids, all_labels, all_preds
        
    """

    convSpeakers, convInputSequence = {}, {}
    convInputMaxSequenceLength = {}
    convActLabels, convEmotionLabels = {}, {}

    conv_ids = set(data['conv_id'])

    for item in list(conv_ids):
        df = data[data.conv_id == item]

        convSpeakers[item] = list(df['speaker'])
        convInputSequence[item] = list(df['sequence'])
        convInputMaxSequenceLength[item] = max(list(df['sentence_length']))
        convActLabels[item] = list(df['encoded_act_label'])
        convEmotionLabels[item] = list(df['encoded_emotion_label'])

    labels, preds = {}, {}
    all_labels, all_preds = [], []
    model.eval()

    for id_ in list(conv_ids):
        # print(f'Making inference for {id_}...')
        data = tensorize(convInputSequence[id_], convSpeakers[id_], convEmotionLabels[id_])

        textf, qmask, umask, label = [d.to(device) for d in data] if cuda_flag else data
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, _, _, _, _ = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        
        preds[id_] = torch.argmax(log_prob, dim=1).cpu().numpy()
        labels[id_] = label.cpu().numpy()

        all_preds.append(preds[id_])
        all_labels.append(labels[id_])

    all_labels, all_preds = np.concatenate(all_labels), np.concatenate(all_preds)

    return labels, preds, list(conv_ids), np.array(all_labels), np.array(all_preds)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--tokenizer', type=str, default='dailydialog/tokenizer.pkl',
                        help='does not use GPU')
    parser.add_argument('--inference-file', type=str, default='inference_sample.json',
                        help='path to json file')
    parser.add_argument('--model-weights', type=str, default='checkpoints/DailyDialog/GraphGRU.pt',
                        help='path to pretrained weights')
    parser.add_argument('--statistics', action='store_true', default=False,
                        help='whether to show statistics')
    parser.add_argument('--show-result', action='store_true', default=False,
                        help='whether to show final result')

    args = parser.parse_args()


    cuda_flag = False
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device('cuda')
        cuda_flag = True
        print(f'Inference on GPU...')
    else:
        device = torch.device('cpu')
        print(f'Inference on CPU...')


    # Loading pre-fitted tokenizer
    if args.tokenizer:
        with open(args.tokenizer, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = False

    # Preparing data
    inference_data, label_decoder = prepare_data(args.inference_file,
                                                 'inference', tokenizer=tokenizer)

    model = DialogueGCN_DailyModel(no_cuda = not cuda_flag)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))

    if cuda_flag:
        model.to(device)

    statistics = args.statistics
    labels, preds, conv_ids, all_labels, all_preds = make_prediction(
        data=inference_data, model=model)

    if statistics:

        accuracy = round(accuracy_score(all_labels, all_preds) * 100, 2)
        fscore = round(f1_score(all_labels, all_preds, average='micro',
                                        labels=[0,1,2,3,4,5,6]) * 100, 2)
        precision = round(precision_score(all_labels, all_preds, average='micro',
                                            labels=[0,1,2,3,4,5,6]) * 100, 2)
        recall = round(recall_score(all_labels, all_preds, average='micro',
                                        labels=[0,1,2,3,4,5,6]) * 100, 2)
        
        print()
        print(f'Average Accuracy: {accuracy}')
        print(f'Average Fscore: {fscore}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print()

    new_data = {'sentences': [],
                'act_label': [],
                'emotion_label': [],
                'predictions': [],
                'speaker': [],
                'conv_id': [],
                'utt_id': []
                }
    
    for id_ in conv_ids:
        df = inference_data[inference_data.conv_id == id_]

        new_data['sentences'] += df.sentence.to_list()
        new_data['act_label'] += df.act_label.to_list()
        new_data['emotion_label'] += df.emotion_label.to_list()
        new_data['speaker'] += df.speaker.to_list()
        new_data['conv_id'] += df.conv_id.to_list()
        new_data['utt_id'] += df.utt_id.to_list()
        new_data['predictions'] += preds[id_].tolist()

    final_data = pd.DataFrame(data=new_data)
    final_data.predictions = final_data.predictions.apply(lambda x: label_decoder[x])

    if args.show_result:
        for i in final_data.index:
            print(f'Sentence: {final_data.loc[i, "sentences"]}')
            print(f'Actual Emotion: {final_data.loc[i, "emotion_label"]}')
            print(f'Predicted Emotion: {final_data.loc[i, "predictions"]}')
            print()
            

