import os
import csv
import numpy as np
import torch
import scipy.io
from torchvision import transforms


class MultiModalDataLoader:
    def __init__(self, config: dict):
        """
        Initialize the MultiModalDataLoader class with the given configuration.
        """
        self.data_path = config['DataPath']['data_path']
        self.stride = config['DataLoader']['seq_stride']
        self.seq_len = config['DataLoader']['seq_length']
        self.imbalance_factor = config['DataLoader']['imbalance_factor']

    def get_recording_idx(self, edf: str) -> int:
        """
        Get the index of the recording from its file name.
        """
        inter = edf.split("_")[-1]
        idx = int(inter[1:])
        return idx

    def load_training_data(self, subjects: list, imbalance_factor: int = 1) -> tuple:
        """
        Load training data for the given subjects.
        """
        for isubj, subject in enumerate(subjects):
            seq_eeg, seq_emg, seq_labels = self.load_data(subject)

            is_seizures = np.sum(seq_labels, axis=-1)
            seizures_index = np.where(is_seizures != 0)[0]
            non_seizures_index = np.where(is_seizures == 0)[0]

            num_seizures = len(seizures_index)
            num_non_seizures = num_seizures * imbalance_factor

            if num_non_seizures > len(non_seizures_index):
                num_non_seizures = len(non_seizures_index)

            np.random.seed(1)
            downsample_non_seizures_index = np.random.choice(non_seizures_index, size=num_non_seizures, replace=False)

            index = seizures_index.tolist() + downsample_non_seizures_index.tolist()

            if isubj == 0:
                train_eeg = seq_eeg[index]
                train_emg = seq_emg[index]
                train_labels = seq_labels[index]
            else:
                train_eeg = np.row_stack((train_eeg, seq_eeg[index]))
                train_emg = np.row_stack((train_emg, seq_emg[index]))
                train_labels = np.row_stack((train_labels, seq_labels[index]))

        return train_eeg, train_emg, train_labels

    def load_subject_data(self, subject: str) -> tuple:
        """
        Load data for a single subject.
        """
        root = os.path.join(self.data_path, subject)
        recordings = self.get_recordings(root)

        for i, recording in enumerate(recordings):
            recording_root = os.path.join(root, recording)
            if i == 0:
                seq_eeg, seq_emg, seq_labels = self.load_recording_data(recording_root)
                seq_eeg_dict = {recording: seq_eeg}
                seq_emg_dict = {recording: seq_emg}
                seq_labels_dict = {recording: seq_labels}
            else:
                seq_eeg, seq_emg, seq_labels = self.load_recording_data(recording_root)
                seq_eeg_dict[recording] = seq_eeg
                seq_emg_dict[recording] = seq_emg
                seq_labels_dict[recording] = seq_labels

        return seq_eeg_dict, seq_emg_dict, seq_labels_dict

    def load_data(self, subject: str) -> tuple:
        """
        Load data for training or validation.
        """
        root = os.path.join(self.data_path, subject)
        recordings = self.get_recordings(root)

        for i, recording in enumerate(recordings):
            recording_root = os.path.join(root, recording)
            if i == 0:
                seq_eeg, seq_emg, seq_labels = self.load_recording_data(recording_root)
            else:
                i_seq_eeg, i_seq_emg, i_seq_labels = self.load_recording_data(recording_root)
                seq_eeg = np.row_stack((seq_eeg, i_seq_eeg))
                seq_emg = np.row_stack((seq_emg, i_seq_emg))
                seq_labels = np.row_stack((seq_labels, i_seq_labels))

        return seq_eeg, seq_emg, seq_labels

    def get_recordings(self, root: str) -> list:
        """
        Get the list of recording files from the specified root directory.
        """
        contents = os.listdir(root)
        files = sorted(contents.copy())

        labels = [file for file in files if 'Labels' in file and '.npy' in file]

        recordings = []
        for label_file in labels:
            ids = label_file.split('_')
            key = '_'.join(ids[0:2])
            recordings.append(key)

        recordings.sort(key=self.get_recording_idx)

        return recordings

    def filter_nan(self, data: np.ndarray) -> np.ndarray:
        """
        Replace NaN values with 0 and infinite values with a large number.
        """
        data[np.isnan(data)] = 0
        data[~np.isfinite(data)] = 10000
        return data

    def load_eeg(self, recording: str) -> np.ndarray:
        """
        Load EEG features from the specified recording.
        """
        name = f"{recording}_Feat_EEG_SD.mat"
        data = scipy.io.loadmat(name)
        eeg_features = data['patient']['EEG_features'][0, 0]
        eeg_features_hf = data['patient']['EEG_features_HF'][0, 0]
        eeg_features_entropy = data['patient']['EEG_features_Entropy'][0, 0]
        features_eeg = np.hstack((eeg_features, eeg_features_hf, eeg_features_entropy))
        features = self.filter_nan(features_eeg)
        return features

    def load_emg(self, recording: str) -> np.ndarray:
        """
        Load EMG features from the specified recording.
        """
        name = f"{recording}_Feat_EMG_SD.npy"
        emg = np.load(name, allow_pickle=True).item()
        features = self.filter_nan(emg['EMG'])
        zeros = np.zeros([features.shape[0], 35])
        features = np.hstack((features, zeros))
        return features

    def load_labels(self, recording: str) -> tuple:
        """
        Load labels from the specified recording.
        """
        name = f"{recording}_Feat_Labels_SD.npy"
        file = np.load(name, allow_pickle=True).item()
        train_labels = file['Labels_Train']
        test_labels = file['Labels_Test']
        ground_truth = file['Labels_Groundtruth']

        test_labels = np.reshape(test_labels, [len(test_labels), 1])
        ground_truth = np.reshape(ground_truth, [len(ground_truth), 1])

        return test_labels, ground_truth

    def load_recording_data(self, recording: str) -> tuple:
        """
        Load data for a single recording.
        """
        seq_len = self.seq_len
        seq_stride = self.stride
        eeg = self.load_eeg(recording)
        emg = self.load_emg(recording)
        labels, _ = self.load_labels(recording)

        seq_eeg = self.timeseries_data_from_array(eeg, sequence_length=seq_len, sequence_stride=seq_stride)
        seq_emg = self.timeseries_data_from_array(emg, sequence_length=seq_len, sequence_stride=seq_stride)
        seq_labels = self.timeseries_data_from_array(labels, sequence_length=seq_len, sequence_stride=seq_stride)

        return seq_eeg, seq_emg, seq_labels

    def timeseries_data_from_array(
        self, 
        data: np.ndarray,
        sequence_length: int,
        sequence_stride: int = 1,
        start_index: int = None,
        end_index: int = None
        ) -> np.ndarray:
        """
        Generate time series data from an array.
        """
        if start_index is not None:
            if start_index < 0:
                raise ValueError(f"`start_index` must be 0 or greater. Received: start_index={start_index}")
            if start_index >= len(data):
                raise ValueError(f"`start_index` must be lower than the length of the data. Received: start_index={start_index}, for data of length {len(data)}")
        if end_index is not None:
            if start_index is not None and end_index <= start_index:
                raise ValueError(f"`end_index` must be higher than `start_index`. Received: start_index={start_index}, and end_index={end_index}")
            if end_index >= len(data):
                raise ValueError(f"`end_index` must be lower than the length of the data. Received: end_index={end_index}, for data of length {len(data)}")
            if end_index <= 0:
                raise ValueError(f"`end_index` must be higher than 0. Received: end_index={end_index}")

        if sequence_stride <= 0:
            raise ValueError(f"`sequence_stride` must be higher than 0. Received: sequence_stride={sequence_stride}")
        if sequence_stride >= len(data):
            raise ValueError(f"`sequence_stride` must be lower than the length of the data. Received: sequence_stride={sequence_stride}, for data of length {len(data)}")

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(data)

        stop_seqs = end_index - start_index - sequence_length + 1
        start_positions = np.arange(start_index, stop_seqs, sequence_stride)
        seq_data = [data[starter:starter + sequence_length] for starter in start_positions]

        return np.array(seq_data)


class MultiModalDataLoading(torch.utils.data.Dataset):
    def __init__(self, eeg: np.ndarray, acc: np.ndarray, emg: np.ndarray, labels: np.ndarray, transform=transforms.ToTensor()):
        """
        Initialize the MultiModalDataLoading class with EEG, ACC, EMG data and labels.
        """
        self.eeg = eeg
        self.acc = acc
        self.emg = emg
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, index: int) -> tuple:
        seq_eeg = self.eeg[index]
        seq_acc = self.acc[index]
        seq_emg = self.emg[index]
        seq_labels = self.labels[index]

        return seq_eeg, seq_acc, seq_emg, seq_labels


class SignalModalDataLoading(torch.utils.data.Dataset):
    def __init__(self, data1: np.ndarray, data2: np.ndarray = None, transform=transforms.ToTensor()):
        """
        Initialize the SignalModalDataLoading class with data.
        """
        self.data1 = data1
        self.data2 = data2

    def __len__(self) -> int:
        return len(self.data1)

    def __getitem__(self, index: int) -> tuple:
        if self.data2 is None:
            return self.data1[index]
        else:
            seq_data1 = self.data1[index]
            seq_data2 = self.data2[index]
            return seq_data1, seq_data2