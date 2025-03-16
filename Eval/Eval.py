import os
import csv
import yaml
import numpy as np
import torch
import scipy.io
from pyedflib import EdfReader
from evaluate_performance import evaluation


class Evaluation:
    def __init__(self, config: dict):
        """
        Initialize the Evaluation class with the given configuration.
        """
        self.data_path = config['data_path']
        self.results_path = config['results_path']
        self.sequence_length = config['seq_length']
        self.stride = config['seq_stride']
        self.test_stride = config['seq_length']
        self.scale = 0.5

    def get_files(self, modality: str, root: str) -> list:
        """
        Get the list of files for the given modality from the specified root directory.
        """
        contents = os.listdir(root)
        files = sorted(contents.copy())
        if modality == 'EEG':
            data = [file for file in files if modality in file and '.mat' in file]
        elif modality == 'ACC':
            data = [file for file in files if modality in file and '.npy' in file and 'EEG' not in file]
        else:
            data = [file for file in files if modality in file and ('.npy' in file or '.pt' in file)]
        return data

    def get_recording_labels(self, root: str, length: int, scale: float) -> tuple:
        """
        Get the labels and ground truth for the recording from the specified root directory.
        """
        annot_file = f"{root}_a2.tsv"
        labels = np.zeros(int(length // scale))

        with open(annot_file) as annotations:
            annot_reader = csv.reader(annotations, delimiter='\t')
            for index, row in enumerate(annot_reader):
                if index > 4 and row[2] == 'Tonic-clonic movements':
                    onset, stop = int(row[0]), int(row[1])
                    if onset > 0:
                        labels[int((onset - 1) // scale):int((stop - 1) // scale)] = 1

        ground_truth_file = f"{root}_a1.tsv"
        ground_truth = np.zeros(int(length // scale))

        with open(ground_truth_file) as annotations:
            annot_reader = csv.reader(annotations, delimiter='\t')
            for index, row in enumerate(annot_reader):
                if index > 4 and row[2] == 'seizure':
                    onset, stop = int(row[0]), int(row[1])
                    if onset > 0:
                        ground_truth[int((onset - 1) // scale):int((stop - 1) // scale)] = 1

        return labels, ground_truth

    def load_results(self, subject: str) -> dict:
        """
        Load the results for the specified subject.
        """
        results = {}
        root = os.path.join(self.results_path, subject)
        results_files = self.get_files('Results', root)

        for file in results_files:
            name = os.path.join(root, file)
            result = torch.load(name)
            key = '_'.join(file.split('_')[0:2])
            results[key] = result

        return results

    def get_recording_length(self, root: str) -> int:
        """
        Get the length of the recording from the specified root directory.
        """
        recording_file = f"{root}.edf"
        with EdfReader(recording_file) as f:
            sample_frequencies = f.getSampleFrequencies()
            num_samples = f.getNSamples()
            lengths = num_samples / sample_frequencies

            if np.max(lengths) == np.min(lengths):
                return int(lengths[0])
            else:
                print('Different channel lengths detected.')
                return int(lengths[0])

    def recording_length(self, subjects_list: list):
        """
        Compute and save the lengths of recordings for all subjects.
        """
        lengths = np.load('RecordingLengths.npy', allow_pickle=True).item()

        for subject in subjects_list:
            root = os.path.join(self.data_path, subject)
            files = [f for f in os.listdir(root) if f.endswith('.edf')]
            for file in files:
                key = file.split('.')[0]
                name = os.path.join(root, key)
                if key not in lengths:
                    length = self.get_recording_length(name)
                    lengths[key] = length

        np.save('RecordingLengths.npy', lengths)

    def tensor_to_array(self, data: torch.Tensor, rec_length: int, scale: float, stride: int = 10) -> np.ndarray:
        """
        Convert a tensor to a NumPy array.
        """
        data = data.detach().numpy()
        pred = np.zeros(int(rec_length // scale))
        starts = np.arange(0, len(data) * int(stride / scale), int(stride / scale))

        for i, start in enumerate(starts):
            pred[start:start + int(stride / scale)] = data[i, :int(stride / scale), 0]

        return pred

    def probs_to_preds(self, probs: torch.Tensor, rec_length: int, scale: float, threshold: float = 0.5) -> np.ndarray:
        """
        Convert probabilities to predictions.
        """
        preds = self.tensor_to_array(probs, rec_length, scale, self.sequence_length)
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0

        return preds

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        """
        Post-process the predictions to smooth the results.
        """
        processed_preds = np.zeros(len(preds))

        for i in range(len(preds)):
            if np.sum(preds[i - 40:i]) >= 40:
                processed_preds[i] = 1
            if processed_preds[i - 200] == 1 and preds[i] == 1:
                processed_preds[i - 200:i] = 1

        return processed_preds

    def save_prediction(self, subject: str, recording: str, preds: np.ndarray):
        """
        Save the predictions for the given subject and recording.
        """
        if np.sum(preds) > 0:
            root = os.path.join(self.data_path, 'Clinical_Review_DL', subject)
            os.makedirs(root, exist_ok=True)
            name = os.path.join(root, f"{recording}_Detections.mat")
            scipy.io.savemat(name, {'Detections': preds})

    def individual_performance(self, subject: str) -> tuple:
        """
        Evaluate the individual performance for the given subject.
        """
        probs = self.load_results(subject)
        scale = self.scale
        recording_lengths = np.load('RecordingLengths.npy', allow_pickle=True).item()
        tp_list, fp_list, fn_list, length_list = [], [], [], []

        for key, prob in probs.items():
            root = os.path.join(self.data_path, subject, key)
            rec_length = recording_lengths[key]
            preds = self.probs_to_preds(prob, rec_length, scale)
            preds = self.postprocess(preds)
            labels, ground_truth = self.get_recording_labels(root, rec_length, scale)
            evals = evaluation(preds, labels, ground_truth, scale)

            tp_list.append(evals['TP'])
            fp_list.append(evals['FP'])
            fn_list.append(evals['FN'])
            length_list.append(rec_length)

        return tp_list, fp_list, fn_list, length_list

    def performance(self, subjects: list) -> tuple:
        """
        Evaluate the overall performance for all subjects.
        """
        tp_list, fp_list, fn_list, length_list = [], [], [], []

        for subject in subjects:
            print(subject)
            i_tps, i_fps, i_fns, i_lengths = self.individual_performance(subject)
            tp_list.append(np.sum(i_tps))
            fp_list.append(np.sum(i_fps))
            fn_list.append(np.sum(i_fns))
            length_list.append(np.sum(i_lengths))
            print(tp_list, np.sum(i_fps) / np.sum(i_lengths) * 3600 * 24)

        return tp_list, fp_list, fn_list, length_list


if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    config_file_path = os.path.join(current_folder, 'Config', 'config.yml')

    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    subjects_list = [
        'SUBJ-1a-025', 'SUBJ-1a-163', 'SUBJ-1a-177', 'SUBJ-1a-224',
        'SUBJ-1a-226', 'SUBJ-1a-349', 'SUBJ-1a-353', 'SUBJ-1a-358',
        'SUBJ-1a-382', 'SUBJ-1a-414', 'SUBJ-1a-434', 'SUBJ-1a-471',
        'SUBJ-1b-178', 'SUBJ-1b-307', 'SUBJ-4-198', 'SUBJ-4-203',
        'SUBJ-4-305', 'SUBJ-4-466', 'SUBJ-5-365', 'SUBJ-6-256',
        'SUBJ-6-275', 'SUBJ-6-276', 'SUBJ-6-291', 'SUBJ-6-357',
        'SUBJ-6-430', 'SUBJ-6-463', 'SUBJ-6-483'
    ]

    test_list = [
        'SUBJ-1a-127', 'SUBJ-1a-188', 'SUBJ-1a-227', 'SUBJ-1a-339',
        'SUBJ-4-139', 'SUBJ-4-169', 'SUBJ-4-265', 'SUBJ-7-282',
        'SUBJ-7-287', 'SUBJ-7-322', 'SUBJ-7-333', 'SUBJ-7-334',
        'SUBJ-7-378', 'SUBJ-7-440', 'SUBJ-7-442', 'SUBJ-7-457'
    ]

    subject_list = subjects_list + test_list

    evaluator = Evaluation(config)
    true_positives, false_positives, false_negatives, lengths = evaluator.performance(subject_list)

    sensitivity = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
    false_positive_rate = np.sum(false_positives) / np.sum(lengths) * 3600 * 24
    positive_predictive_value = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    f1_score = 2 * positive_predictive_value * sensitivity / (positive_predictive_value + sensitivity)

    results_file = 'DL.mat'
    scipy.io.savemat(results_file, {'Metrics': [true_positives, false_positives, false_negatives, lengths]})
    print(sensitivity, false_positive_rate, positive_predictive_value, f1_score)