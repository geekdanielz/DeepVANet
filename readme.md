# DeepVANet

The PyTorch implementation for the paper 'DeepVANet: A Deep End-to-End Network for Multi-modal Emotion Recognition'.    

## Dependencies
+ Python 3.7
+ PyTorch 1.5.0
+ torchvision 0.6.0
+ numpy 1.17.2
+ pandas 1.1.2
+ opencv-python 4.4.0.42
+ Pillow 7.2.0

## Instructions
* Two public datasets are used in this paper: DEAP\[1\] and MAHNOB-HCI\[2\]. Please access the datasets via:    
DEAP: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html    
MAHNOB-HCI: http://mahnob-db.eu/hci-tagging    

* data_preprocess.py contains the functions used for data pre-process. It also provides a preprocess_demo() to preprocess DEAP dataset.
After running preprocess_demo.py, the face and bio-sensing data of each subject should be compressed to .zip format.    
The final organization should be like follows:    
./data/    
　　-- DEAP/    
　　　　-- faces/    
　　　　　-- s{subject_id}.zip    
　　　　-- bio/    
　　　　　-- s{subject_id}.zip    
　　　　-- labels/    
　　　　　-- participant_ratings.csv    
　　-- MAHNOB/    
　　　　-- faces/    
　　　　　-- s{subject_id}.zip    
　　　　-- bio/    
　　　　　-- s{subject_id}.zip    
　　　　-- labels/    
　　　　　-- mahnob_labels.npy    

* Run demo.py to train and test the model using per-subject experiments.    
Arguments for demo.py:    

| Arguments| Description | Default |
|---|---|---|
| --dataset | The dataset used for evaluation | DEAP |
| --modal | Data modality | facebio |
| --label | Emotional label | valence |
| --subject | Subject id | 1 |
| --fusion | Fusion strategy| feature|
| --epoch | The number of epochs in training| 50 |
| --batch_size | The batch size used in training | 64 |
| --learn_rate | Learn rate in training| 0.001 |
| --face_feature_size | Face feature size | 16 |
| --bio-feature_size | Bio-sensing feature size| 64 |
| --gpu | Use gpu or not | True |
| --pretrain | Use pretrained CNN | True |


## References
\[1\] Koelstra, S., Muhl, C., Soleymani, M., Lee, J.S., Yazdani, A., Ebrahimi, T., Pun,
T., Nijholt, A., Patras, L.: Deap: A database for emotion analysis using physiolog-
ical signals. IEEE Transactions on Affective Computing 3(1), 18–31 (2012)    
\[2\] Soleymani, M., Lichtenauer, J., Pun, T., Pantic, M.: A multimodal database for affect recognition and implicit tagging. IEEE Transactions on Affective Computing 3(1), 42–55 (2012)

## Citation
Zhang Y., Hossain M.Z., Rahman S. (2021) DeepVANet: A Deep End-to-End Network for Multi-modal Emotion Recognition. In: Ardito C. et al. (eds) Human-Computer Interaction – INTERACT 2021. INTERACT 2021. Lecture Notes in Computer Science, vol 12934. Springer, Cham. https://doi.org/10.1007/978-3-030-85613-7_16
