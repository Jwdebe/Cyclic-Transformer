import os
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from pyedflib import EdfReader
from sklearn.preprocessing import StandardScaler
from Data.sTransform import sTransform


class Signals:
    def __init__(self, config: dict) -> None:
        """
        Initialize the Signals class with the given configuration.
        """
        self.data_path = config['data_path']
        self.training_data_path = config['training_data_path']
        self.sequence_length = config['seq_length']
        self.stride = config['seq_stride']
        self.imbalance_factor = config['imbalance_factor']
        self.scale = config['scale']

    def load_training_data(self, subjects_list: list, *modalities: str) -> tuple:
        """
        Load training data for the given subjects.

        Args:
            subjects_list (list): List of subjects.
            modalities (str): Modalities to be loaded.

        Returns:
            tuple: A tuple containing signals and labels.
        """
        imbalance_factor = self.imbalance_factor
        signals = {modality: [] for modality in modalities}

        for isubj, subject in enumerate(subjects_list):
            name = os.path.join(self.training_data_path, subject, f"{subject}_TrainingData.npy")
        
            isignals = np.load(name, allow_pickle=True).item()
            ilabels = isignals['TrainLabels']

            if isubj == 0:
                labels = ilabels
            else:
                labels = np.row_stack((labels, ilabels))

            for modality in modalities:
                if isubj == 0:
                    signals[modality] = isignals[modality]
                else:
                    signals[modality] = np.row_stack((signals[modality], isignals[modality]))

        return signals, labels
    

    def load_subject_data(self, subject: str, *args: str) -> dict:
        """
        Load data for a single subject.
        """

        name = os.path.join(self.training_data_path, subject, f"{subject}_Testing.npy")
        
        signals = np.load(name, allow_pickle=True).item()

        return signals