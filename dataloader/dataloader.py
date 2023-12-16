import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from aeon.datasets import load_classification
from sklearn.preprocessing import MinMaxScaler
from itertools import chain

from sktime.transformations.panel.dictionary_based import SFA
from aeon.datasets import load_classification
from transformers import AutoModel
from transformers import RobertaTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import pywt
from torch.nn import functional as F
from tqdm import tqdm
import joblib


tokenizer = RobertaTokenizer.from_pretrained("allenai/longformer-base-4096")


multiv = [
    'MotorImagery',
     'InsectWingbeat',
 'DuckDuckGeese',
 'ArticularyWordRecognition',
 'SelfRegulationSCP1',
 'Cricket',
 'PhonemeSpectra',
"SpokenArabicDigits",
 'BasicMotions',
 'Epilepsy',
 'PEMS-SF',
 'ERing',
 'SelfRegulationSCP2',
 'Heartbeat',
 'Libras',
 'FingerMovements',
 'AtrialFibrillation',
 'UWaveGestureLibrary',
 'NATOPS',
 'EthanolConcentration',
 'FaceDetection',
 'StandWalkJump',
 'PenDigits',
 'Handwriting',
 'RacketSports',
 'LSST',
 'HandMovementDirection',
 'CharacterTrajectories',
 'EigenWorms',
 'JapaneseVowels']
univ = ['Wine',
 'CricketX',
#  'CharacterTrajectories',
 'UWaveGestureLibraryZ',
 'Fish',
 'ECG5000',
 'Lightning7',
 'FaceAll',
 'FordA',
 'ItalyPowerDemand',
 'GunPointMaleVersusFemale',
 'CinCECGTorso',
 'Earthquakes',
 'ShapesAll',
 'GunPoint',
 'SonyAIBORobotSurface2',
 'Worms',
 'MixedShapesSmallTrain',
 'EOGVerticalSignal',
 'DodgerLoopDay',
 'Yoga',
 'Symbols',
 'PowerCons',
 'ECG200',
 'Herring',
 'Meat',
 'PigArtPressure',
 'ProximalPhalanxTW',
 'Adiac',
 'UWaveGestureLibraryAll',
 'WordSynonyms',
 'MoteStrain',
 'Fungi',
 'MiddlePhalanxOutlineAgeGroup',
 'Car',
 'CBF',
 'ProximalPhalanxOutlineCorrect',
 'DodgerLoopWeekend',
 'SmallKitchenAppliances',
 'Lightning2',
 'FreezerRegularTrain',
 'GunPointAgeSpan',
 'UWaveGestureLibraryX',
 'UWaveGestureLibraryY',
 'ElectricDevices',
 'BME',
 'MiddlePhalanxTW',
 'DiatomSizeReduction',
 'Plane',
 'Computers',
 'InsectEPGRegularTrain',
 'Coffee',
 'Rock',
 'Beef',
 'Phoneme',
 'MedicalImages',
 'HandOutlines',
 'SemgHandGenderCh2',
 'LargeKitchenAppliances',
 'ACSF1',
 'ToeSegmentation2',
 'StarLightCurves',
 'NonInvasiveFetalECGThorax2',
 'InlineSkate',
 'ShapeletSim',
 'BirdChicken',
 'WormsTwoClass',
 'DistalPhalanxOutlineCorrect',
 'FiftyWords',
 'Trace',
 'InsectEPGSmallTrain',
 'FordB',
 'EOGHorizontalSignal',
 'Mallat',
 'SmoothSubspace',
 'ScreenType',
 'FacesUCR',
 'PigCVP',
 'Wafer',
 'NonInvasiveFetalECGThorax1',
 'MiddlePhalanxOutlineCorrect',
 'SemgHandSubjectCh2',
 'SemgHandMovementCh2',
 'DistalPhalanxOutlineAgeGroup',
 'PigAirwayPressure',
 'Chinatown',
 'FreezerSmallTrain',
 'SonyAIBORobotSurface1',
 'FaceFour',
 'DistalPhalanxTW',
 'ChlorineConcentration',
 'ECGFiveDays',
 'SyntheticControl',
 'BeetleFly',
 'CricketZ',
 'PhalangesOutlinesCorrect',
 'DodgerLoopGame',
 'EthanolLevel',
 'Strawberry',
 'UMD',
 'MixedShapesRegularTrain',
 'Ham',
 'OliveOil',
 'Crop',
 'ProximalPhalanxOutlineAgeGroup',
 'Haptics',
 'TwoLeadECG',
 'TwoPatterns',
 'ArrowHead',
 'ToeSegmentation1',
 'CricketY',
 'HouseTwenty',
 'RefrigerationDevices',
 'GunPointOldVersusYoung',
 'SwedishLeaf',
 'OSULeaf']
univ_set = set(univ)
multiv_set = set(multiv)

remove_set = {'PEMS-SF',
    'PenDigits',
    'MotorImagery',
    'EigenWorms',
    'DuckDuckGeese',
    'DodgerLoopGame',
    'SmoothSubspace',
    'FiftyWords',
    'DiatomSizeReduction',
    'Phoneme',
    'DodgerLoopDay',
    'Fungi',
    'DodgerLoopWeekend',
    }
multiv_set = set(multiv)
univ_set = set(univ)


SMAP_entities = ['P-1','S-1','E-1','E-2','E-3','E-4','E-5','E-6','E-7','E-8','E-9','E-10'
 'E-11','E-12','E-13','A-1','D-1','P-2','P-3','D-2','D-3','D-4','A-2'
 'A-3','A-4','G-1','G-2','D-5','D-6','D-7','F-1','P-4','G-3','T-1','T-2'
 'D-8','D-9','F-2','G-4','T-3','D-11','D-12','B-1','G-6','G-7','P-7','R-1'
 'A-5','A-6','A-7','D-13','P-2','A-8','A-9','F-3']

MSL_entities = ['M-6','M-1','M-2','S-2','P-10','T-4','T-5','F-7','M-3','M-4','M-5','P-15'
 'C-1','C-2','T-12','T-13','F-4','F-5','D-14','T-9','P-14','T-8','P-11'
 'D-15','D-16','M-7','F-8']

SMD_entities = ['machine-1-1', 'machine-1-6', 'machine-1-7',
                        'machine-2-1', 'machine-2-2', 'machine-2-7', 'machine-2-8',
                        'machine-3-3', 'machine-3-4', 'machine-3-6', 'machine-3-8', 
                        'machine-3-11'][6:]
SMD_entities = ['machine-1-7']

ASD_entities = ['omi-' + str(i) for i in range(1, 13)]  

ad_entities = {"SMAP": SMAP_entities,
               "MSL": MSL_entities,
               "SMD": SMD_entities,
               "ASD": ASD_entities}

dataset_names = sorted(list((multiv_set | univ_set) - remove_set))
interested_d_set = set([
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions",
    "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms",
    "Epilepsy", "EthanolConcentration", "ERing", "FaceDetection",
    "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat",
    "InsectWingbeat", "JapaneseVowels", "Libras", "LSST", "MotorImagery",
    "NATOPS", "PenDigits", "PEMS-SF", "Phoneme", "RacketSports",
    "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
    "StandWalkJump", "UWaveGestureLibrary", 'PhonemeSpectra'
])

def get_ad_data(dataset, channel, window_size):
    DATA_PATH = './AD_data'
    if dataset == 'SMAP' or dataset == 'MSL':
        train = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'train', channel + '.npy'))
        test = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'test', channel + '.npy'))
        
        label = np.load(os.path.join(DATA_PATH, 'SMAP&MLS', 'labels', channel + '.npy'))[window_size - 1:]
        
    elif dataset == 'SMD' or dataset == 'ASD':
        train = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD', channel + '_train.pkl'))
        test = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD', channel + '_test.pkl'))
        
        raw_label = joblib.load(os.path.join(DATA_PATH, 'SMD&ASD', channel + '_test_label.pkl'))[window_size - 1:]

    label = np.ones_like(raw_label, dtype=np.int32)
    label[raw_label == 1] = -1
    train = np.nan_to_num(train, 0)
    test = np.nan_to_num(test, 0)

    scaler = MinMaxScaler((-1, 1)).fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    train_data = torch.from_numpy(train).unfold(0, window_size, 1).numpy()
    test_data = torch.from_numpy(test).unfold(0, window_size, 1).numpy()
    if train_data.shape[-1] < 224:
        train_data = torch.nn.functional.interpolate(torch.tensor(train_data), (224, ))
        test_data = torch.nn.functional.interpolate(torch.tensor(test_data), (224, ))
    return {'samples': train_data,'labels': label,'class_num':2}, {'samples': test_data,'labels': label,'class_num':2}

# def get_data(dataset_name, split):
#     if dataset_name in univ_set:
#         extract_path = "/home/user1/liangchen/aeon/aeon/datasets/data/test/Univariate_ts"
#     if dataset_name in multiv_set:
#         extract_path = "/home/user1/liangchen/aeon/aeon/datasets/data/test/Multivariate_ts"
#     X, y, meta = load_classification(dataset_name, split=split, extract_path=extract_path)
#     class_values = meta["class_values"]
#     y_ = np.zeros(y.shape, dtype=int)
#     for idx, cv in enumerate(class_values):
#         y_ += (cv == y) * idx
#     mean = np.nanmean(X)
#     std = np.nanstd(X)
#     X = (X - mean) / std
#     if X.shape[-1] < 224:
#         X = torch.nn.functional.interpolate(torch.tensor(X), (224, ))
#         X = X.numpy()
#     return {"samples":torch.from_numpy(X), "labels":torch.from_numpy(y_), "class_num":len(np.unique(y))}

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                # print(line)
                # exit()
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict

def get_data_and_label_from_ts_file(file_path,label_dict):
    # print(file_path)
    # exit()
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y
    
def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    
def get_data(dataset_name, split="train"):
    if dataset_name in univ_set:
        extract_path = "/home/user1/liangchen/aeon/aeon/datasets/data/test/Univariate_ts"
    if dataset_name in multiv_set:
        extract_path = "/home/user1/liangchen/aeon/aeon/datasets/data/test/Multivariate_ts"
    Train_dataset_path = extract_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = extract_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    desired_dim = 144
    if split == "train":
        X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path,label_dict)
        X, y_ = set_nan_to_zero(X_train), y_train
        mean = np.nanmean(X)
        std = np.nanstd(X)
        X = (X - mean) / std

        if X.shape[-1] > 17000:
            X = X[...,:3000]
            # X = torch.nn.functional.interpolate(torch.tensor(X), (5000, ))
            # X = X.numpy()
        
        if X.shape[-2] > desired_dim:
            X = torch.nn.functional.adaptive_avg_pool2d(torch.tensor(X), (desired_dim, X.shape[-1]))
            X = X.numpy()

        if X.shape[-1] < 112:
            X = torch.nn.functional.interpolate(torch.tensor(X), (112, ), mode='linear')
            X = X.numpy()
        return {"samples":torch.from_numpy(X), "labels":torch.from_numpy(y_), "class_num":len(label_dict)}
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path,label_dict)
    X, y_ = set_nan_to_zero(X_test), y_test 
    mean = np.nanmean(X)
    std = np.nanstd(X)
    X = (X - mean) / std

    if X.shape[-1] > 17000:
        X = X[...,:3000]
        # X = torch.nn.functional.interpolate(torch.tensor(X), (5000, ))
        # X = X.numpy()
    if X.shape[-2] > desired_dim:
        X = torch.nn.functional.adaptive_avg_pool2d(torch.tensor(X), (desired_dim, X.shape[-1]))
        X = X.numpy()
    if X.shape[-1] < 112:
        X = torch.nn.functional.interpolate(torch.tensor(X), (112, ), mode='linear')
        X = X.numpy()
    # print(torch.from_numpy(X).shape)
    # exit()
    return {"samples":torch.from_numpy(X), "labels":torch.from_numpy(y_), "class_num":len(label_dict)}

def list2string(word):
    string_ = ""
    for w in word:
        string_ += str(w)
    return string_
# print(results)

def wavelet_features(X, d_name, resize_shape=224, dset_type="train", wavelet="db3"):

    save_path = f"augmentation/wave_{d_name}_{resize_shape}_{wavelet}_{dset_type}.pt"
    coeffs = []
    if os.path.exists(save_path):
        return torch.load(save_path)
    train_size, channel, n = X.shape
    scale = resize_shape // 2
    scales = np.arange(1, scale + 1, 1)
    # X = X.reshape((train_size * channel, n))
    print("wavelet trainsform to: " + save_path)
    for d in tqdm(X):
        d_ = []
        for dd in d:
            coef, freqs = pywt.cwt(dd, scales, wavelet)
            if resize_shape > n:
                resize_shape = resize_shape // 2
            coef = F.interpolate(torch.abs(torch.tensor(coef)).unsqueeze(0).unsqueeze(0), size=(resize_shape, resize_shape), 
                        mode='bilinear', align_corners=False).squeeze()
            d_.append(coef)
        coef = torch.stack(d_)
        desired_channels = 64
        # print(coef.shape)
        if channel > desired_channels:
            coef = coef.view(1, coef.shape[0], resize_shape * resize_shape)
            coef = F.adaptive_avg_pool2d(coef, (desired_channels, resize_shape * resize_shape))
            coef = coef.view(desired_channels, resize_shape, resize_shape)
        coeffs.append(coef)
    print("stacking")
    coeffs = torch.stack(coeffs)
    print("finally stacked")
    # coeffs = coeffs.view((train_size, channel, resize_shape , resize_shape))
    # coeffs = torch.tensor(coeffs)
    print("saving")
    torch.save(coeffs, save_path)
    print("final saved")
    return coeffs
    
def sfa_features(X, d_name, max_channel=20, dset_type="train"):
    sfa = SFA()
    sfas = []
    space = tokenizer.tokenize(" ")[0]
    save_path = f"augmentation/sfa_{d_name}_{max_channel}_{dset_type}.pt"
    if os.path.exists(save_path):
        return torch.load(save_path)
    else:
        for b in range(X.shape[0]):
            tokens_all = []
            for d in range(min(X.shape[1], max_channel)):
                results = sfa.fit_transform(X[b][d][None, None])[0][0]
                prompt_info = []
                for word, num in results.items():
                    prompt_info.append(
                        (sfa.word_list_typed(word), num)
                    )
                prompt = f"The most frequent words in the bag for the {d}th channel are: " 
                prompt_info = sorted(prompt_info, key=lambda x:x[1], reverse=True)

                words_tokens = []
                for word, num in prompt_info[:min(5, len(prompt_info))]:
                    word = list2string(word)
                    words_tokens.append(",")
                    words_tokens.append(space)
                    words_tokens.append(str(num))
                    words_tokens.append(space)
                    words_tokens += word
                words_tokens.append('.')
                words_tokens = words_tokens[2:]

                tokens = tokenizer.tokenize(prompt) + words_tokens
                tokens_all.append(space)
                tokens_all += tokens
            sfas.append(tokens_all)

        tokenized = []
        for sf in sfas:
            sf = tokenizer.encode(sf, return_tensors="pt")
            tokenized.append(sf[0])
        sfas = pad_sequence(tokenized, batch_first=True, padding_value=tokenizer.encode("<PAD>")[0])
        torch.save(sfas, save_path)
        return sfas

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, d_name=None, dset_type="test"):
        super(Load_Dataset, self).__init__()
        self.d_name = d_name
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # self.sfas = sfa_features(X_train.numpy(), d_name, dset_type=dset_type)
        print(d_name)

        if "ad" in training_mode or \
            d_name == "InsectWingbeat" or dset_type=="test":
            self.sfas = X_train # sfa_features(X_train.numpy(),d_name, dset_type=dset_type)
            self.db1 = self.sfas # wavelet_features(X_train.numpy(), d_name, dset_type=dset_type, wavelet="db1")
            self.coif5 = self.sfas # wavelet_features(X_train.numpy(), d_name, dset_type=dset_type, wavelet="coif5")
            self.dmey = self.sfas # wavelet_features(X_train.numpy(), d_name, dset_type=dset_type, wavelet="dmey")
        else:
            self.sfas = sfa_features(X_train.numpy(),d_name, dset_type=dset_type)
            self.db1 = wavelet_features(X_train.numpy(), d_name, dset_type=dset_type, wavelet="db1")
            self.coif5 = wavelet_features(X_train.numpy(), d_name, dset_type=dset_type, wavelet="coif5")
            self.dmey = wavelet_features(X_train.numpy(), d_name, dset_type=dset_type, wavelet="dmey")


        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train
        

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index], self.sfas[index], self.db1[index], self.coif5[index], self.dmey[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode, window_size=100):
    if data_path == 'all':
        d_names = dataset_names
        # print("Using dsets: " + str(d_names))
    else:
        d_names = [data_path]
    
    chain_dataloaders = [dict() for _ in range(3)]
    d_name_use = []
    for d in d_names:
        if 'ad' not in training_mode:
            train_dataset = get_data(d, "train")
        # print(train_dataset["samples"].shape)
        # exit()
            test_dataset = get_data(d, "test")
        else:
            dash = d.find('-')
            dataset = d[:dash]
            channel = d[dash + 1:]
            # print(dataset, channel)
            # exit()
            train_dataset, test_dataset = get_ad_data(dataset, channel, window_size)
            d = d + "-" + str(window_size)

        valid_dataset = test_dataset

        len_train = train_dataset["samples"].shape[0]
        # print(train_dataset["samples"].shape)
        # exit()
        len_test = test_dataset["samples"].shape[0]
        configs.input_channels = train_dataset["samples"].shape[1]
        configs.num_classes = test_dataset["class_num"]
        configs.input_length = train_dataset["samples"].shape[2]

        print("dataset name: " + d)
        print("train set length: " + str(len_train))
        print("test set length: " + str(len_test))
        print("input_length: " + str(configs.input_length))
        print("channel: " + str(train_dataset["samples"].shape[1]))
        # if configs.input_length < 48:
        #     if len(d_names) == 1:
        #         raise RuntimeError(d + ", input_lenght: " + str(configs.input_lenght))
        #     continue
        print()

        # train_dataset = torch.load(os.path.join(data_path, "train.pt"))
        # valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
        # test_dataset = torch.load(os.path.join(data_path, "test.pt"))
        # print(train_dataset)
        # raise RuntimeError
        train_dataset = Load_Dataset(train_dataset, configs, training_mode, d_name=d, dset_type="train")
        valid_dataset = Load_Dataset(valid_dataset, configs, training_mode, d_name=d, dset_type="test")
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, d_name=d, dset_type="test")

        # print(len(train_dataset))
        # exit()


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                shuffle=True, drop_last=True,
                                                num_workers=0)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                                shuffle=False, drop_last=False,
                                                num_workers=0)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                                shuffle=False, drop_last=False,
                                                num_workers=0)
        chain_dataloaders[0][d]=train_loader
        chain_dataloaders[1][d]=valid_loader
        chain_dataloaders[2][d]=test_loader
        d_name_use.append(d)
    # chain_dataloaders = [chain(cd)  for cd in chain_dataloaders]
    train_loader = chain_dataloaders[0][d_name_use[0]]
    valid_loader = chain_dataloaders[1]
    test_loader = chain_dataloaders[2]


    return train_loader, valid_loader, test_loader