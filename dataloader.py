import torch
import pickle
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class IEMOCAPDataset(Dataset):
    """
    Class to load the IEMOCAP dataset.

    Args:
        train: whether to load data for training, Default: True
        
    """

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainvid if train else self.testvid)]

        self.len = len(self.keys)


    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid


    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(data[i]) if i<4 else pad_sequence(dat[i], True)
                if i<6 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):
    """
    Class to load daily dialogue dataset.

    Args:
        split: train/ valid/ test
        path: path to preprocessed daily dialogue dataset pickle file
        
    """

    def __init__(self, split, path):
        self.Speakers, self.Features, _, self.ActLabels, self.EmotionLabels, \
                       self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.validId]
        elif split == 'valid':
            self.keys = [x for x in self.testId]

        self.len = len(self.keys)


    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.FloatTensor(self.Features[conv]), \
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1]*len(self.EmotionLabels[conv])), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               conv


    def __len__(self):
        return self.len


    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        return [pad_sequence(dat[i]) if i<2 else pad_sequence(dat[i], True)
                if i<4 else dat[i].tolist() for i in dat]
