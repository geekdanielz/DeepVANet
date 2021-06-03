"""
Functions for data pre-process
"""

import cv2
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from PIL import Image
import _pickle as cPickle
import os
import face_alignment
from xml.dom.minidom import parse
import pyedflib
import shutil


# Stimulation time (second) of trials elicited by different videos in MAHNOB-HCI datasets; keys are the names of stimulation videos, values are the lengths of videos.
# Note, videos in MAHNOB-HCI dataset include extra time for participant self-assessment, so we only extract frames during the stimulation.
trial_time = {'69.avi': 58,
              '55.avi': 76,
              '58.avi': 58,
              'earworm_f.avi': 53,
              '53.avi':	104,
              '80.avi':	96,
              '52.avi':	96,
              '79.avi':	42,
              '73.avi':	71,
              '90.avi':	85,
              '107.avi': 34,
              '146.avi': 87,
              '30.avi':	70,
              '138.avi': 116,
              'newyork_f.avi': 89,
              '111.avi': 113,
              'detroit_f.avi': 89,
              'cats_f.avi': 98,
              'dallas_f.avi': 89,
              'funny_f.avi': 87
              }


def MAHNOB_summary():
    '''
    This function extracts information (trialID, subjectID, stimulation time, valence, arousal) from each trials in MAHNOB-HCI dataset.
    Note, when preprocessing MAHNOB-HCI dataset, this function should be called before video2frames().
    '''
    root = './datasets/MAHNOB/Sessions/'
    data = []
    for trial in os.listdir(root):
        if not trial[0] == '.':
            file = root + trial + '/session.xml'
            try:
                xml = parse(file)
                r = xml.documentElement
                arousal = r.getAttribute('feltArsl')
                valence = r.getAttribute('feltVlnc')
                media = r.getAttribute('mediaFile')
                sub = r.getElementsByTagName('subject')[0].getAttribute('id')
                data.append([int(trial), int(sub), int(trial_time[media]), int(valence), int(arousal)])
            except:
                print(f'Information of trial {trial} is incomplete.')
    arr = np.array(data)
    np.save('./data/MAHNOB/labels/mahnob_labels.npy', arr)


# ************************* Face Data Pre-process *************************

def video2frames(dataset='DEAP'):
    '''
    Extract frames from videos.
    :param dataset: used dataset
    '''
    assert dataset in ['DEAP', 'MAHNOB'], 'Invalid dataset name'

    if dataset == 'DEAP':
        dataset_path = './datasets/DEAP/face_video/'
        des_path = './datasets/DEAP/frames/'
        for subject in os.listdir(dataset_path):
            if subject.startswith('.'):
                continue
            sub_path = dataset_path+subject
            for video_file in os.listdir(sub_path):
                if not os.path.exists(des_path + subject):
                    os.mkdir(des_path + subject)
                if not os.path.exists(des_path + subject + '/' + video_file.split('.')[0]):
                    os.mkdir(des_path + subject + '/' + video_file.split('.')[0])
                video_file_path = sub_path+'/'+video_file
                video = cv2.VideoCapture(video_file_path)
                c = 1
                frame_rate = 10
                count = 0
                while (True):
                    ret, frame = video.read()
                    if ret:
                        if (c % frame_rate == 0):
                            count += 1
                            cv2.imwrite(des_path+subject+'/'+video_file.split('.')[0] +'/'+ video_file.split('.')[0]+'_'+str(count) + '.png', frame)
                        c += 1
                        cv2.waitKey(0)
                    else:
                        break
                video.release()

    if dataset == 'MAHNOB':
        dataset_path = './datasets/MAHNOB/Sessions/'
        des_path = './datasets/MAHNOB/frames/'
        labels = np.load('./data/MAHNOB/labels/mahnob_labels.npy')
        for l in labels:
            trial = l[0]
            subject = l[1]
            time = l[2]
            for video_file in os.listdir(dataset_path+str(trial)):
                if video_file.endswith('.avi'):
                    video_file_path = dataset_path + str(trial) +'/' + video_file
                    video = cv2.VideoCapture(video_file_path)
                    if not os.path.exists(des_path + str(subject)):
                        os.mkdir(des_path + str(subject))
                    if not os.path.exists(des_path + str(subject) +'/' + str(trial)):
                        os.mkdir(des_path + str(subject) + '/' + str(trial))
                    c = 1
                    frame_rate = 12
                    count = 0
                    while (True):
                        if count > time * 5:
                            break
                        ret, frame = video.read()
                        if ret:
                            if (c % frame_rate == 0):
                                count += 1
                                cv2.imwrite(
                                    des_path + str(subject) + '/' + str(trial) + '/' + str(trial) + '_'+ str(
                                        count) + '.png', frame)
                            c += 1
                            cv2.waitKey(0)
                        else:
                            break
                    video.release()

# functions for face alignment and cropping are based on https://github.com/DANNALI35/zhihu_article/tree/master/201901_face_alignment
def to_dict(landmarks):
    '''
    Transfer detected facial landmarks list to dictionary.
    :param landmarks: a list of facial landmarks
    :return: a dictionary of facial landmarks
    '''
    l = list()
    for i in range(68):
        point = (landmarks[i][0], landmarks[i][1])
        l.append(point)
    face_landmarks_dict = dict()
    face_landmarks_dict['chin'] = l[0:17]
    face_landmarks_dict['left_eyebrow'] = l[17:22]
    face_landmarks_dict['right_eyebrow'] = l[22:27]
    face_landmarks_dict['nose_bridge'] = l[27:31]
    face_landmarks_dict['nose_tip'] = l[31:36]
    face_landmarks_dict['left_eye'] = l[36:42]
    face_landmarks_dict['right_eye'] = l[42:48]
    face_landmarks_dict['top_lip'] = l[48:55] + l[60:65]
    face_landmarks_dict['bottom_lip'] = l[55:60] + l[65:68]
    return face_landmarks_dict


def crop_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 18 / 40
    bottom = lip_center[1] + mid_part * 12 / 40

    w = h = bottom - top
    x_center = eye_center[0]
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


def align_landmarks(landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
#     rotated_landmarks = defaultdict(list)
    rotated_landmarks = []
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=570)
#             rotated_landmarks[facial_feature].append(rotated_landmark)
            rotated_landmarks.append(rotated_landmark)
    return rotated_landmarks


def face_detection_alignment_cropping(dataset='DEAP'):
    '''
    Transfer frames to faces by face detection, alignment and cropping.
    :param dataset: used dataset
    '''
    assert dataset in ['DEAP', 'MAHNOB'], 'Invalid dataset name'

    # facial landmarks detector; use gpu by changing device parameter to 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

    if dataset == 'DEAP':
        root = './datasets/DEAP/frames/'
        des_path = './data/DEAP/faces/'

    if dataset == 'MAHNOB':
        root = './datasets/MAHNOB/frames/'
        des_path = './data/DEAP/faces/'

    for subject in os.listdir(root):
        for trial in os.listdir(root+subject):
            if os.path.exists(des_path + subject + '/' + trial):
                continue
            os.mkdir(des_path + subject + '/' + trial)
            for frame in os.listdir(root+subject+'/'+trial):
                frame_path = root + subject + '/' + trial + '/' + frame
                img = cv2.imread(frame_path)
                preds = fa.get_landmarks(img)
                try:
                    landmarks_list = preds[0]
                    landmarks_dict = to_dict(landmarks_list)
                    aligned_face, eye_center, angle = align_face(image_array=img, landmarks=landmarks_dict)
                    rotated_landmarks = rotate_landmarks(landmarks=landmarks_dict, eye_center=eye_center, angle=angle,
                                                     row=img.shape[0])
                    cropped_img, left, top = crop_face(image_array=aligned_face, landmarks=rotated_landmarks)

                    cv2.imwrite(des_path + subject + '/' + trial + '/' + frame, cropped_img)
                except:
                    print(f'Fail to get the face image: {frame}')


# ************************* Bio-sensing Data Pre-process *************************

def trial2segments(dataset='DEAP'):
    '''
    Divide bio-sensing data of each trial to 1-second length segments, and perform baseline removal.
    Note, when dealing with MAHNOB-HCI dataset, EEG data should be common reference averaged, bandpass filtered and artefact removed using EEGLab,
    and preprocessed EEG data files (one file per trial) should be stored in './datasets/MAHNOB/eeg_preprocessed/ folder in .npy format.
    :param dataset: used dataset
    '''
    assert dataset in ['DEAP', 'MAHNOB'], 'Invalid dataset name'

    if dataset == 'DEAP':
        root = './datasets/DEAP/data_preprocessed_python/'
        des_path = './data/DEAP/bio/'
        labels = pd.read_csv('./data/DEAP/labels/participant_ratings.csv')
        for file in os.listdir(root):
            subject = file.split('.')[0]
            sub_id = int(subject[1:])
            os.mkdir(des_path + 's' + str(sub_id))
            f = open(root + file, 'rb')
            d = cPickle.load(f, encoding='latin1')
            data = d['data']
            for experiment in range(40):
                trial = labels[(labels['Participant_id'] == sub_id) & (labels['Experiment_id'] == experiment + 1)][
                    'Trial'].iloc[0]
                # baseline
                l = []
                for i in range(3):
                    l.append(data[experiment][:, i * 128:(i + 1) * 128])
                baseline_mean = sum(l) / 3
                # segments
                for i in range(60):
                    data_seg = data[experiment][:, 384 + i * 128:384 + (i + 1) * 128]
                    data_seg_removed = data_seg - baseline_mean
                    np.save(f'{des_path}s{sub_id}/{sub_id}_{trial}_{i + 1}.npy', data_seg_removed)

    if dataset == 'MAHNOB':
        root = './datasets/MAHNOB/Sessions/'
        eeg_root = './datasets/MAHNOB/eeg_preprocessed/'
        des_path = '/data/MAHNOB/bio/'
        indeces = [32, 33, 34, 40, 44, 45] # used bio-sensing data channel indeces
        labels = np.load('./data/MAHNOB/labels/mahnob_labels.npy')
        for i in range(len(labels)):
            trial = labels[i][0]
            subject = labels[i][1]
            time = labels[i][2]
            for file in os.listdir(f'{root}{trial}'):
                if file.endswith('.bdf'):
                    with pyedflib.EdfReader(f'{root}{trial}/file') as f:
                        channels = []
                        for index in indeces:
                            channel = np.zeros(f.samples_in_file(index), dtype='float64')
                            f.readsignal(index, 0, f.samples_in_file(index), channel)
                            channel = channel[27 * 256:(30 + time) * 256:2].reshape(1, -1)
                            channels.append(channel)
                        peri = np.concatenate(channels, 0)
                        eeg = np.load(f'{eeg_root}{trial}.npy').T[:, 27 * 128:(30 + time) * 128]
                        bio = np.concatenate([eeg, peri], 0)
                        baseline1 = bio[:, :128]
                        baseline2 = bio[:, 128:256]
                        baseline3 = bio[:, 256:384]
                        baseline_mean = (baseline1 + baseline2 + baseline3) / 3
                        for segment in range(time):
                            data = bio[:, (3 + segment) * 128:(4 + segment) * 128]
                        data = data - baseline_mean
                        if not os.path.exists(f'{des_path}{subject}/'):
                            os.mkdir(f'{des_path}{subject}/')
                        np.save(f'{des_path}{subject}/{subject}_{trial}_{segment + 1}.npy', data)



def preprocess_demo():
    '''
    This function pre-processes DEAP dataset.
    Please unzip face_video.zip, data_preprocessed_python.zip and metadata_csv.zip from DEAP dataset in './datasets/DEAP/'.
    Then call this function, the preprocessed data will be stored in './data/DEAP/'.
    It is recommended to use a device with GPU, otherwise the face detection will be slow.
    Note that faces cannot be detected from some frames, these frames should be replaced with the neighbour frame manually.
    '''
    # pre-process face data
    if not os.path.exists('./datasets/DEAP/frames/'):
        os.mkdir('./datasets/DEAP/frames/')
    if not os.path.exists('./data/'):
        os.mkdir('./data/')
    if not os.path.exists('./data/DEAP/'):
        os.mkdir('./data/DEAP/')
    if not os.path.exists('./data/DEAP/faces/'):
        os.mkdir('./data/DEAP/faces/')
    if not os.path.exists('./data/DEAP/labels/'):
        os.mkdir('./data/DEAP/labels/')
    shutil.copy('./datasets/DEAP/metadata_csv/participant_ratings.csv', './data/DEAP/labels/participant_ratings.csv')
    video2frames('DEAP')
    face_detection_alignment_cropping('DEAP')

    # preprocess bio-sensing data
    if not os.path.exists('./data/DEAP/bio/'):
        os.mkdir('./data/DEAP/bio/')
    trial2segments('DEAP')


if __name__ == '__main__':
    preprocess_demo()