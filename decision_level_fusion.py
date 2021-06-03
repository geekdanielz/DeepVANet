"""
Decision-level fusion functions
"""

from dataset import DEAP, MAHNOB
from train import train
from models import DeepVANetVision, DeepVANetBio, DeepVANet
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import out_put
import os


def adaboost(face_model_path, bio_model_path, bio_type, indices, dataset='DEAP', subject=1, k=1, label='valence', use_gpu=False):
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # sub-classifiers
    face_model = DeepVANetVision().to(device)
    face_model.load(face_model_path)
    if dataset == 'DEAP':
        if bio_type == 'bio': bio_input_size = 40
        if bio_type == 'eeg': bio_input_size = 32
        if bio_type == 'peri': bio_input_size = 8
    if dataset == 'MAHNOB':
        if bio_type == 'bio': bio_input_size = 48
        if bio_type == 'eeg': bio_input_size = 32
        if bio_type == 'peri': bio_input_size = 6
    bio_model = DeepVANetBio(bio_input_size).to(device)
    bio_model.load(bio_model_path)

    if dataset == 'DEAP':
        face_train = DEAP(subject=subject, modal='face', label=label, kind='train', indices=indices, k=k)
        face_test = DEAP(subject=subject, modal='face', label=label, kind='val', indices=indices, k=k)
        bio_train = DEAP(subject=subject, modal=bio_type, label=label, kind='train', indices=indices, k=k)
        bio_test = DEAP(subject=subject, modal=bio_type, label=label, kind='val', indices=indices, k=k)
    if dataset == 'MAHNOB':
        face_train = MAHNOB(subject=subject, modal='face', label=label, kind='train', indices=indices, k=k)
        face_test = MAHNOB(subject=subject, modal='face', label=label, kind='val', indices=indices, k=k)
        bio_train = MAHNOB(subject=subject, modal=bio_type, label=label, kind='train', indices=indices, k=k)
        bio_test = MAHNOB(subject=subject, modal=bio_type, label=label, kind='val', indices=indices, k=k)

    n_train = len(face_train)
    n_test = len(face_test)

    face_train_loader = DataLoader(face_train, batch_size=64, shuffle=False)
    face_test_loader = DataLoader(face_test, batch_size=64, shuffle=False)
    bio_train_loader = DataLoader(bio_train, batch_size=64, shuffle=False)
    bio_test_loader = DataLoader(bio_test, batch_size=64, shuffle=False)

    # initialize weights
    w = np.ones(n_train) / n_train

    face_pred = []
    face_train_y = []
    for ii, (x, y) in enumerate(face_train_loader):
        print(ii)
        pred = face_model(x.to(device)).cpu().detach().numpy()
        face_pred.append(pred)
        face_train_y.append(y.detach().numpy())
    face_pred = np.concatenate(face_pred)
    face_train_y = np.concatenate(face_train_y)
    train_accuracy_face = sum((face_pred>=0.5).astype(float) == face_train_y) / n_train
    I = (face_pred==face_train_y).astype(float)
    I2 = np.array([1 if x==1 else -1 for x in I])
    error = sum(abs(face_pred-face_train_y)*w)
    alpha_face = 0.5 * np.log((1 - error)/error)
    w_updated = np.multiply(w, np.exp([float(x) * alpha_face for x in I2]))

    bio_pred = []
    bio_train_y = []
    for ii, (x, y) in enumerate(bio_train_loader):
        print(ii)
        pred = bio_model(x.to(device)).cpu().detach().numpy()
        bio_pred.append(pred)
        bio_train_y.append(y.detach().numpy())
    bio_pred = np.concatenate(bio_pred)
    bio_train_y = np.concatenate(bio_train_y)
    train_accuracy_bio = sum((bio_pred >= 0.5).astype(float) == bio_train_y) / n_train
    I = (bio_pred == bio_train_y).astype(float)
    I2 = np.array([1 if x == 1 else -1 for x in I])
    error = sum(abs(bio_pred - bio_train_y) * w_updated)
    alpha_bio = 0.5 * np.log((1 - error)/error)

    face_s = []
    face_test_y = []
    for ii, (x, y) in enumerate(face_test_loader):
        print(ii)
        pred = face_model(x.to(device)).cpu().detach().numpy()
        face_s.append(pred)
        face_test_y.append(y.detach().numpy())
    face_s = np.concatenate(face_s)
    face_test_y = np.concatenate(face_test_y)

    bio_s = []
    for ii, (x, y) in enumerate(bio_test_loader):
        print(ii)
        pred = bio_model(x.to(device)).cpu().detach().numpy()
        bio_s.append(pred)
    bio_s = np.concatenate(bio_s)

    face_s_mapped = 2 * face_s - 1
    bio_s_mapped = 2 * bio_s - 1

    final_score = 1 / (1 + np.exp(-(alpha_face*face_s_mapped+alpha_bio*bio_s_mapped)))

    final_pred = (final_score>=0.5).astype(float)
    test_accuracy = sum(final_pred==face_test_y) / n_test

    print(f'train accuracy face: {train_accuracy_face}, train accuracy bio: {train_accuracy_bio}, alpha_face: {alpha_face}, alpha_bio: {alpha_bio}, test accuracy: {test_accuracy}')
    return train_accuracy_face, train_accuracy_bio, alpha_face, alpha_bio, test_accuracy


def decision_fusion(dataset, modal, subject, k, label, indices, use_gpu, pretrain=True):
    # train sub-classifiers
    bio_modal = modal[4:]
    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')
    if not os.path.exists(f'./results/{dataset}/'):
        os.mkdir(f'./results/{dataset}/')
    if not os.path.exists(f'./results/{dataset}/face/'):
        os.mkdir(f'./results/{dataset}/face/')
    if not os.path.exists(f'./results/{dataset}/{bio_modal}/'):
        os.mkdir(f'./results/{dataset}/{bio_modal}/')
    train(modal='face', dataset=dataset, epoch=50, lr=0.001, use_gpu=use_gpu,
               file_name=f'./results/{dataset}/face/{dataset}_decision_f_{label}_s{subject}_k{k}/{dataset}_decision_f_{label}_s{subject}_k{k}',
               batch_size=64, subject=subject, k=k, l=label, indices=indices,pretrain=pretrain)
    train(modal=bio_modal, dataset=dataset, epoch=50, lr=0.001, use_gpu=use_gpu,
              file_name=f'./results/{dataset}/{bio_modal}/{dataset}_decision_{bio_modal}_{label}_s{subject}_k{k}/{dataset}_decision_{bio_modal}_{label}_s{subject}_k{k}',
              batch_size=64, subject=subject, k=k, l=label, indices=indices)
    face_model_path = f'./results/{dataset}/face/{dataset}_decision_f_{label}_s{subject}_k{k}/{dataset}_decision_f_{label}_s{subject}_k{k}.pth'
    bio_model_path = f'./results/{dataset}/{bio_modal}/{dataset}_decision_{bio_modal}_{label}_s{subject}_k{k}/{dataset}_decision_{bio_modal}_{label}_s{subject}_k{k}.pth'

    train_accuracy_face, train_accuracy_bio, alpha_face, alpha_bio, test_accuracy = adaboost(face_model_path, bio_model_path, bio_modal,indices, dataset, subject, k, label, use_gpu)
    out_put(f'train accuracy face: {train_accuracy_face}, train accuracy bio: {train_accuracy_bio}, alpha_face: {alpha_face}, alpha_bio: {alpha_bio}, test accuracy: {test_accuracy}', f'./results/{dataset}/{modal}/{dataset}_decision_{label}_s{subject}_k{k}_{modal}')




