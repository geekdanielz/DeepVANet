"""
DEAP and MAHNOB datasets
"""

import torch
import os
from PIL import Image
from torch.utils import data
import numpy as np
import pandas as pd
from torchvision import transforms as T
import io
import zipfile


class DEAP(data.Dataset):
    '''
    DEAP dataset for per-subject experiments

    Parameters:
        modal   : data modality; {'face', 'eeg', 'peri', 'bio', 'faceeeg', 'faceperi', 'facebio'}, default = 'facebio'.
        subject : subject ID; an integer between 1 and 22, default = 1.
        k       : kth fold; an integer between 1 and 10, default = 1.
        kind    : dataset type; {'train', 'val', 'all'}, default = 'all'.
        indices : index of data samples (for dataset shuffle); an list of integers, default = list(range(2400)).
        label   : emotion label; {'valence', 'arousal'}, defualt = 'valence'.
    '''
    def __init__(self, modal='facebio', subject=1, k=1, kind='all', indices=list(range(2400)),label='valence'):
        self.modal = modal
        self.subject = subject
        self.k = k
        self.kind = kind
        self.label = label
        self.bio_path = f'./data/DEAP/bio/s{subject}.zip'
        self.label_path = f'./data/DEAP/labels/'
        self.face_path = f'./data/DEAP/faces/s{subject}.zip'
        self.labels = pd.read_csv(self.label_path+'participant_ratings.csv')
        self.face_zip = zipfile.ZipFile(self.face_path, 'r')
        self.bio_zip = zipfile.ZipFile(self.bio_path, 'r')
        self.size = len(indices)

        if kind == 'train':
            self.indices = indices[:int((k - 1) * self.size / 10)] + indices[int(k * self.size / 10):]
        if kind == 'val':
            self.indices = indices[int((k - 1) * self.size / 10):int(k * self.size / 10)]
        if kind == 'all':
            self.indices = indices

    def __getitem__(self, i):
        index = self.indices[i]
        trial = index // 60 + 1
        segment = index % 60 + 1
        prex = 's' + (str(self.subject) if self.subject > 9 else '0' + str(self.subject)) + '/s' + (
            str(self.subject) if self.subject > 9 else '0' + str(self.subject)) + '_trial' + (
                   str(trial) if trial > 9 else '0' + str(trial)) + '/s' + (
                   str(self.subject) if self.subject > 9 else '0' + str(self.subject)) + '_trial' + (
                   str(trial) if trial > 9 else '0' + str(trial))
        transform = T.Compose([T.Resize((64, 64)),
                               T.ToTensor()])

        face_data = []
        for n in range(1, 6):
            img = Image.open(io.BytesIO(self.face_zip.read(prex + f'_{(segment - 1) * 5 + n}.png')))
            frame_array = transform(img)
            frame_array = frame_array.view(1, 3, 64, 64)
            face_data.append(frame_array)
        face_data = torch.cat(face_data, dim=0)
        bio_data = torch.tensor(
            np.load(io.BytesIO(self.bio_zip.read(f's{self.subject}/{self.subject}_{trial}_{segment}.npy')))).float()

        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'peri':
            data = bio_data[32:]
        elif self.modal == 'bio':
            data = bio_data
        elif self.modal == 'faceeeg':
            data = (face_data, bio_data[:32])
        elif self.modal == 'faceperi':
            data = (face_data, bio_data[32:])
        elif self.modal == 'facebio':
            data = (face_data, bio_data)

        valence = 0 if self.labels[(self.labels['Participant_id']==self.subject) & (self.labels['Trial']==trial)]['Valence'].iloc[0] < 5 else 1
        arousal = 0 if self.labels[(self.labels['Participant_id']==self.subject) & (self.labels['Trial']==trial)]['Arousal'].iloc[0] < 5 else 1
        if self.label == 'valence':
            return data, valence
        else:
            return data, arousal

    def __len__(self):
        return len(self.indices)


class MAHNOB(data.Dataset):
    '''
    MAHNOB-HCI dataset for per-subject experiments

    Parameters:
        modal   : data modality; {'face', 'eeg', 'peri', 'bio', 'faceeeg', 'faceperi', 'facebio'}, default = 'facebio'.
        subject : subject ID; an integer between 1 and 22, default = 1.
        k       : kth fold; an integer between 1 and 10, default = 1.
        kind    : dataset type; {'train', 'val', 'all'}, default = 'all'.
        indices : index of data samples (for dataset shuffle); an list of integers, default = list(range(2400)).
        label   : emotion label; {'valence', 'arousal'}, defualt = 'valence'.
    '''
    def __init__(self, modal='facebio', subject=1, k=1, kind='all', indices=list(range(1611)), label='valence'):
        self.modal = modal
        self.subject = subject
        self.k = k
        self.kind = kind
        self.label = label
        self.bio_path = f'./data/MAHNOB/bio/s{subject}.zip'
        self.label_path = './data/MAHNOB/labels/mahnob_labels.npy'
        self.face_path = f'./data/MAHNOB/faces/s{subject}.zip'
        self.labels = np.load(self.label_path)
        self.label_dict = dict()
        for label in self.labels:
            self.label_dict[label[0]] = label[1:]
        self.trials = []
        self.face_zip = zipfile.ZipFile(self.face_path,'r')
        self.bio_zip = zipfile.ZipFile(self.bio_path, 'r')
        for i in self.face_zip.namelist():
            if not int(i.split('/')[1]) in self.trials:
                self.trials.append(int(i.split('/')[1]))
        self.trials.sort()
        self.trial_seg = []
        for trial in self.trials:
            for seg in range(1,self.label_dict[trial][1]+1):
                self.trial_seg.append((trial,seg))
        self.size = len(indices)

        if kind == 'train':
            self.indices = indices[:int((k - 1) * self.size / 10)] + indices[int(k * self.size / 10):]
        if kind == 'val':
            self.indices = indices[int((k - 1) * self.size / 10):int(k * self.size / 10)]
        if kind == 'all':
            self.indices = indices

    def __getitem__(self, i):
        index = self.indices[i]
        trial, segment = self.trial_seg[index]
        transform = T.Compose([T.Resize((64, 64)),
                               T.ToTensor()])
        face_data = []
        for n in range(1, 6):
            img = Image.open(
                io.BytesIO(self.face_zip.read(f'{self.subject}/{trial}/{trial}_{(segment - 1) * 5 + n}.png')))
            frame_array = transform(img)
            frame_array = frame_array.view(1, 3, 64, 64)
            face_data.append(frame_array)
        face_data = torch.cat(face_data, dim=0)

        bio_data = torch.tensor(np.load(io.BytesIO(self.bio_zip.read(f'{self.subject}/{self.subject}_{trial}_{segment}.npy')))).float()


        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'peri':
            data = bio_data[32:]
        elif self.modal == 'bio':
            data = bio_data
        elif self.modal == 'faceeeg':
            data = (face_data, bio_data[:32])
        elif self.modal == 'faceperi':
            data = (face_data, bio_data[32:])
        elif self.modal == 'facebio':
            data = (face_data, bio_data)
        valence = 0 if self.label_dict[trial][2] < 5 else 1
        arousal = 0 if self.label_dict[trial][3] < 5 else 1
        if self.label == 'valence':
            return data, valence
        else:
            return data, arousal

    def __len__(self):
        return len(self.indices)


class DEAPAll(data.Dataset):
    def __init__(self, modal='facebio', k=1, kind='all', indices=list(range(52440)),label='valence'):
        self.modal = modal
        self.k = k
        self.kind = kind
        self.label = label
        self.bio_path = f'./data/DEAP/bio/'
        self.label_path = f'./data/DEAP/labels/'
        self.face_path = f'./data/DEAP/faces/'
        self.labels = pd.read_csv(self.label_path+'participant_ratings.csv')
        deap_indices_dict = {1: 2400,
                             2: 2400,
                             3: 2340,
                             4: 2400,
                             5: 2340,
                             6: 2400,
                             7: 2400,
                             8: 2400,
                             9: 2400,
                             10: 2400,
                             11: 2220,
                             12: 2400,
                             13: 2400,
                             14: 2340,
                             15: 2400,
                             16: 2400,
                             17: 2400,
                             18: 2400,
                             19: 2400,
                             20: 2400,
                             21: 2400,
                             22: 2400}
        self.sub_trial_seg = []
        for sub in range(1,23):
            for trial in range(1,int(deap_indices_dict[sub]/60+1)):
                for seg in range(1,61):
                    self.sub_trial_seg.append((sub,trial,seg))
        self.size = len(indices)

        if kind == 'train':
            self.indices = indices[:int((k - 1) * self.size / 10)] + indices[int(k * self.size / 10):]
        if kind == 'val':
            self.indices = indices[int((k - 1) * self.size / 10):int(k * self.size / 10)]
        if kind == 'all':
            self.indices = indices

    def __getitem__(self, i):
        index = self.indices[i]
        subject, trial, segment = self.sub_trial_seg[index]
        face_zip = zipfile.ZipFile(self.face_path+f's{subject}.zip', 'r')
        bio_zip = zipfile.ZipFile(self.bio_path+f's{subject}.zip', 'r')
        prex = 's' + (str(subject) if subject > 9 else '0' + str(subject)) + '/s' + (
            str(subject) if subject > 9 else '0' + str(subject)) + '_trial' + (
                   str(trial) if trial > 9 else '0' + str(trial)) + '/s' + (
                   str(subject) if subject > 9 else '0' + str(subject)) + '_trial' + (
                   str(trial) if trial > 9 else '0' + str(trial))
        transform = T.Compose([T.Resize((64, 64)),
                               T.ToTensor()])
        face_data = []
        for n in range(1, 6):
            img = Image.open(io.BytesIO(face_zip.read(prex + f'_{(segment - 1) * 5 + n}.png')))
            frame_array = transform(img)
            frame_array = frame_array.view(1, 3, 64, 64)
            face_data.append(frame_array)
        face_data = torch.cat(face_data, dim=0)

        bio_data = torch.tensor(np.load(io.BytesIO(bio_zip.read(f's{subject}/{subject}_{trial}_{segment}.npy')))).float()

        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'peri':
            data = bio_data[32:]
        elif self.modal == 'bio':
            data = bio_data
        elif self.modal == 'faceeeg':
            data = (face_data, bio_data[:32])
        elif self.modal == 'faceperi':
            data = (face_data, bio_data[32:])
        elif self.modal == 'facebio':
            data = (face_data, bio_data)

        valence = 0 if self.labels[(self.labels['Participant_id']==subject) & (self.labels['Trial']==trial)]['Valence'].iloc[0] < 5 else 1
        arousal = 0 if self.labels[(self.labels['Participant_id']==subject) & (self.labels['Trial']==trial)]['Arousal'].iloc[0] < 5 else 1
        if self.label == 'valence':
            return data, valence
        else:
            return data, arousal

    def __len__(self):
        return len(self.indices)


class MAHNOBAll(data.Dataset):
    def __init__(self, modal='facebio', k=1, kind='all', indices=list(range(42463)), label='valence'):
        self.modal = modal
        self.k = k
        self.kind = kind
        self.label = label
        self.bio_path = f'./data/MAHNOB/bio/'
        self.label_path = './data/MAHNOB/labels/mahnob_labels.npy'
        self.face_path = f'./data/MAHNOB/faces/'
        self.labels = np.load(self.label_path)
        self.label_dict = dict()
        mahnob_indices_dict = {1: 1611,
                               2: 1611,
                               3: 1305,
                               4: 1611,
                               5: 1611,
                               6: 1611,
                               7: 1611,
                               8: 1611,
                               9: 1124,
                               10: 1611,
                               11: 1611,
                               13: 1611,
                               14: 1611,
                               16: 1370,
                               17: 1611,
                               18: 1611,
                               19: 1611,
                               20: 1611,
                               21: 1611,
                               22: 1611,
                               23: 1611,
                               24: 1611,
                               25: 1611,
                               27: 1611,
                               28: 1611,
                               29: 1611,
                               30: 1611}
        for label in self.labels:
            self.label_dict[label[0]] = label[1:]
        self.sub_trial_seg = []
        for sub in mahnob_indices_dict:
            for i in os.listdir(self.bio_path+str(sub)):
                if i.endswith('.npy'):
                    trial = int(i.split('.')[0].split('_')[1])
                    seg = int(i.split('.')[0].split('_')[2])
                    self.sub_trial_seg.append((sub,trial,seg))

        self.size = len(indices)

        if kind == 'train':
            self.indices = indices[:int((k - 1) * self.size / 10)] + indices[int(k * self.size / 10):]
        if kind == 'val':
            self.indices = indices[int((k - 1) * self.size / 10):int(k * self.size / 10)]
        if kind == 'all':
            self.indices = indices

    def __getitem__(self, i):
        index = self.indices[i]
        subject, trial, segment = self.sub_trial_seg[index]
        face_zip = zipfile.ZipFile(self.face_path + f's{subject}.zip', 'r')
        bio_zip = zipfile.ZipFile(self.bio_path + f's{subject}.zip', 'r')
        transform = T.Compose([T.Resize((64, 64)),
                               T.ToTensor()])

        face_data = []
        for n in range(1, 6):
            img = Image.open(io.BytesIO(face_zip.read(f'{subject}/{trial}/{trial}_{(segment - 1) * 5 + n}.png')))
            frame_array = transform(img)
            frame_array = frame_array.view(1, 3, 64, 64)
            face_data.append(frame_array)
        face_data = torch.cat(face_data, dim=0)

        bio_data = torch.tensor(np.load(io.BytesIO(bio_zip.read(f'{subject}/{subject}_{trial}_{segment}.npy')))).float()

        if self.modal == 'face':
            data = face_data
        elif self.modal == 'eeg':
            data = bio_data[:32]
        elif self.modal == 'peri':
            data = bio_data[32:]
        elif self.modal == 'bio':
            data = bio_data
        elif self.modal == 'faceeeg':
            data = (face_data, bio_data[:32])
        elif self.modal == 'faceperi':
            data = (face_data, bio_data[32:])
        elif self.modal == 'facebio':
            data = (face_data, bio_data)

        valence = 0 if self.label_dict[trial][2] < 5 else 1
        arousal = 0 if self.label_dict[trial][3] < 5 else 1
        if self.label == 'valence':
            return data, valence
        else:
            return data, arousal

    def __len__(self):
        return len(self.indices)