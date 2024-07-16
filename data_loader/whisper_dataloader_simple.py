import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.logger import logger
from os.path import join
from os import listdir

class WhisperDataset(Dataset):
    def __init__(self, filepaths: list, labels: list):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        whisper_feature = torch.load(filepath)
        # Difference from WhisperDataset in whisper_dataloader.py
        whisper_feature = whisper_feature.mean(axis=1)
        label = self.labels[idx]
        return whisper_feature, label


def CollateFunction(data):
    """
       data: is a list of tuples with (feature, score)
             where 'feature' is a tensor of arbitrary shape
             and score is a scalar
    """
    features = []
    labels = []
    for feature, label in data:
        features.append(feature)
        labels.append(label)

    features = pad_sequence([f.squeeze() for f in features], batch_first=True)
    labels = torch.tensor(labels)
    return features, labels


class WhisperDataloaderSimple(DataLoader):
    def __init__(self, data_dir, train_filelist, test_filelist, val_filelist, train_batch_size, test_batch_size, val_batch_size, shuffle=False):
        self.test_filelist = test_filelist
        self.val_filelist = val_filelist
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir

        wavs_dir = join(self.data_dir)

        with open(join(self.data_dir, train_filelist)) as f:
            train_data = f.readlines()

        train_filepaths = []
        train_targets = []
        for line in train_data:
            filename, label = line.strip().split(",")
            train_filepaths.append(join(wavs_dir, filename))
            train_targets.append(float(label))

        logger.info("Train dataset: {}  files loaded".format(len(train_filepaths)))

        self.dataset = WhisperDataset(train_filepaths, train_targets)

        super().__init__(dataset=self.dataset, batch_size=self.train_batch_size, shuffle=self.shuffle, num_workers=0, collate_fn=CollateFunction)

    def get_val_dataloader(self, set="val" ):
        wavs_dir = join(self.data_dir)

        if set=='val':
            with open(join(self.data_dir, self.val_filelist)) as f:
                val_data = f.readlines()
        else:
            with open(join(self.data_dir, self.test_filelist)) as f:
                val_data = f.readlines()

        val_filepaths = []
        val_targets = []
        for line in val_data:
            filename, label = line.strip().split(",")
            val_filepaths.append(join(wavs_dir, filename))
            val_targets.append(float(label))


        logger.info("{} dataset: {} files loaded".format('Validation' if set == 'val' else 'Testing', len(val_filepaths)))

        self.val_dataset = WhisperDataset(val_filepaths, val_targets)
        batch_size = self.val_batch_size if set == 'val' else self.test_batch_size
        return DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=CollateFunction)