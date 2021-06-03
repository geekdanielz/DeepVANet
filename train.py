"""
Train and test
"""
from torchnet import meter
import torch
import os
from torch.utils.data import DataLoader
from models import DeepVANetBio, DeepVANetVision, DeepVANet
from dataset import DEAP, MAHNOB, DEAPAll, MAHNOBAll
from utils import out_put


def train(modal, dataset, subject, k, l, epoch, lr, batch_size, file_name, indices, face_feature_size=16, bio_feature_size=64, use_gpu=False, pretrain=True):
    '''
    Train and test the model. Output the results.
    :param modal: data modality
    :param dataset: used dataset
    :param subject: subject id
    :param k: kth fold
    :param l: emotional label
    :param epoch: the number of epoches
    :param lr: learn rate
    :param batch_size: training batach size
    :param file_name: result file name
    :param indices: a list of index of the dataset
    :param face_feature_size: face feature size
    :param bio_feature_size: bio-sensing feature size
    :param use_gpu: use gpu or not
    :param pretrain: use pretrained cnn nor not
    :return: the best test accuracy
    '''
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    directory = file_name.split('/')[-2]
    if not os.path.exists(f'./results/{dataset}/{modal}/'+directory):
        os.mkdir(f'./results/{dataset}/{modal}/'+directory)

    if dataset == 'DEAP':
        ############## inter-subjects ##############
        if subject == 0:
            train_data = DEAPAll(modal=modal, k=k, kind='train', indices=indices, label=l)
            val_data = DEAPAll(modal=modal, k=k, kind='val', indices=indices, label=l)
        ############## per-subjects ##############
        else:
            train_data = DEAP(modal=modal,subject=subject,k=k,kind='train',indices=indices, label=l)
            val_data = DEAP(modal=modal,subject=subject,k=k,kind='val',indices=indices, label=l)
        bio_input_size = 40
        peri_input_size = 8
    if dataset == 'MAHNOB':
        ############## inter-subjects  ##############
        if subject == 0:
            train_data = MAHNOBAll(modal=modal, k=k, kind='train', indices=indices, label=l)
            val_data = MAHNOBAll(modal=modal, k=k, kind='val', indices=indices, label=l)
        ############## per-subject #################
        else:
            train_data = MAHNOB(modal=modal,subject=subject,k=k,kind='train',indices=indices, label=l)
            val_data = MAHNOB(modal=modal,subject=subject,k=k,kind='val',indices=indices, label=l)
        bio_input_size = 38
        peri_input_size = 6

    # model
    if modal == 'face':
        model = DeepVANetVision(feature_size=face_feature_size,pretrain=pretrain).to(device)
    if modal == 'bio':
        model = DeepVANetBio(input_size=bio_input_size, feature_size=bio_feature_size).to(device)
    if modal == 'eeg':
        model = DeepVANetBio(input_size=32, feature_size=bio_feature_size).to(device)
    if modal == 'peri':
        model = DeepVANetBio(input_size=peri_input_size, feature_size=bio_feature_size).to(device)
    if modal == 'faceeeg':
        model = DeepVANet(bio_input_size=32, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, pretrain=pretrain).to(device)
    if modal == 'faceperi':
        model = DeepVANet(bio_input_size=peri_input_size, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, pretrain=pretrain).to(device)
    if modal == 'facebio':
        model = DeepVANet(bio_input_size=bio_input_size, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, pretrain=pretrain).to(device)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # criterion and optimizer
    criterion = torch.nn.BCELoss()
    lr = lr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # meters
    loss_meter = meter.AverageValueMeter()

    best_accuracy = 0
    best_epoch = 0

    # train
    for epoch in range(epoch):
        pred_label = []
        true_label = []

        loss_meter.reset()
        for ii, (data,label) in enumerate(train_loader):
            # print(ii)
            # train model
            if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                input = (data[0].float().to(device), data[1].float().to(device))
            else:
                input = data.float().to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            pred = model(input).float()
            # print(pred.shape,label.shape)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            # meters update
            loss_meter.add(loss.item())

            pred = (pred >= 0.5).float().to(device).data
            pred_label.append(pred)
            true_label.append(label)

        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)

        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy: ' + str(train_accuracy.item()), file_name)

        val_accuracy = val(modal, model, val_loader, use_gpu)

        out_put('Epoch: ' + 'train' + str(epoch) + '| train loss: ' + str(loss_meter.value()[0]) +
              '| val accuracy: ' + str(val_accuracy.item()), file_name)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            model.save(f"{file_name}_best.pth")

    model.save(f'{file_name}.pth')

    perf = f"best accuracy is {best_accuracy} in epoch {best_epoch}" + "\n"
    out_put(perf,file_name)

    return best_accuracy


@torch.no_grad()
def val(modal, model, dataloader, use_gpu):
    model.eval()
    if use_gpu:
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    pred_label = []
    true_label = []

    for ii, (data, label) in enumerate(dataloader):
        if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
            input = (data[0].float().to(device), data[1].float().to(device))
        else:
            input = data.float().to(device)
        label = label.to(device)
        pred = model(input).float()

        pred = (pred >= 0.5).float().to(device).data
        pred_label.append(pred)
        true_label.append(label)

    pred_label = torch.cat(pred_label, 0)
    true_label = torch.cat(true_label, 0)

    val_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)

    model.train()

    return val_accuracy
