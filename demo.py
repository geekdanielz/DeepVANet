"""
Demo for per-subject experiment
"""
import train
from train import train
from decision_level_fusion import decision_fusion
import random
import argparse
import os

# deap subject_id: sample number
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
             22: 2400,
             23: 2400,
             24: 2400,
             25: 2400,
             26: 2400,
             27: 2400,
             28: 2400,
             29: 2400,
             30: 2400,
             31: 2400,
             32: 2400}

# mahnob subject_id: sample number
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

def demo():
    parser = argparse.ArgumentParser(description='Per-subject experiment')
    parser.add_argument('--dataset', '-d', default='DEAP', help='The dataset used for evaluation', type=str)
    parser.add_argument('--fusion', default='feature', help='Fusion strategy (feature or decision)', type=str)
    parser.add_argument('--epoch', '-e', default=50, help='The number of epochs in training', type=int)
    parser.add_argument('--batch_size', '-b', default=64, help='The batch size used in training', type=int)
    parser.add_argument('--learn_rate', '-l', default=0.001, help='Learn rate in training', type=float)
    parser.add_argument('--gpu', '-g', default='True', help='Use gpu or not', type=str)
    # parser.add_argument('--file', '-f', default='./results/results.txt', help='File name to save the results', type=str)
    parser.add_argument('--modal', '-m', default='facebio', help='Type of data to train', type=str)
    parser.add_argument('--subject', '-s', default=1, help='Subject id', type=int)
    parser.add_argument('--face_feature_size', default=16, help='Face feature size', type=int)
    parser.add_argument('--bio_feature_size', default=64, help='Bio feature size', type=int)
    parser.add_argument('--label', default='valence', help='Valence or arousal', type=str)
    parser.add_argument('--pretrain',default='True', help='Use pretrained CNN', type=str)

    args = parser.parse_args()

    use_gpu = True if args.gpu == 'True' else False
    pretrain = True if args.pretrain == 'True' else False

    if args.dataset == 'DEAP':
        indices = list(range(deap_indices_dict[args.subject]))
    if args.dataset == 'MAHNOB':
        indices = list(range(mahnob_indices_dict[args.subject]))
    # shuffle the dataset
    random.shuffle(indices)

    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')
    if not os.path.exists(f'./results/{args.dataset}/'):
        os.mkdir(f'./results/{args.dataset}/')
    if not os.path.exists(f'./results/{args.dataset}/{args.modal}/'):
        os.mkdir(f'./results/{args.dataset}/{args.modal}/')

    for k in range(1, 11):
        if args.fusion == 'feature':
            train(modal=args.modal, dataset=args.dataset, epoch=args.epoch, lr=args.learn_rate, use_gpu=use_gpu,
                        file_name=f'./results/{args.dataset}/{args.modal}/{args.dataset}_{args.modal}_{args.label}_s{args.subject}_k{k}_{args.face_feature_size}_{args.bio_feature_size}/{args.dataset}_{args.modal}_{args.label}_s{args.subject}_k{k}_{args.face_feature_size}_{args.bio_feature_size}',
                        batch_size=args.batch_size, subject=args.subject, k=k, l=args.label, indices=indices,
                        face_feature_size=args.face_feature_size, bio_feature_size=args.bio_feature_size, pretrain=pretrain)
        if args.fusion == 'decision':
            decision_fusion(args.dataset, args.modal, args.subject, k, args.label, indices, use_gpu, pretrain)


if __name__ == '__main__':
    demo()