import os
import random
import numpy as np
import torch
import yaml
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Data.MultiModalDataLoader import MultiModalDataLoading, SingalModalDataLoading
from Data.Signals import Signals
from Model.CyclicTransformer import Translator
from Train.Cyclic import Model


class MultiModal():
    def __init__(self, config) -> None:

        self.SD = 2
        self.config = config
        self.results_path = config['results_path']
        self.classifier_path = config['classifier_path']
        self.save_root = config['classifier_path']
        self.batch_size = config['batch_size']
        self.scale = config['scale']
        self.seed = self._set_seed(2048)
        self.generator = self._get_generator(2048)
        self.DataLoader = Signals(config)

    def _set_seed(self, seed):
        work_seed = torch.initial_seed()
        torch.manual_seed(seed)        
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _get_generator(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)

        return g

    def standard_scale(self, X_train):
        preprocessor = StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)

        return X_train, preprocessor

    def minmax_scale(self, X_train):
        preprocessor = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = preprocessor.transform(X_train)

        return X_train, preprocessor       

    def standardize(self, Data, Preprocesser = None):
        
        Seqlen = Data.shape[1]
        Numfeats = Data.shape[2]

        ProcessInput = np.reshape(Data, [Data.shape[0] * Seqlen, Numfeats]) 
        if Preprocesser is None:
            ProcessedData, Preprocesser = self.standard_scale(ProcessInput)
        else:
            ProcessedData = Preprocesser.transform(ProcessInput)

        return np.reshape(ProcessedData, [-1, Seqlen, Numfeats]), Preprocesser

    def minmax(self, Data, Preprocesser = None):
        
        Seqlen = Data.shape[1]
        Numfeats = Data.shape[2]

        ProcessInput = np.reshape(Data, [Data.shape[0] * Seqlen, Numfeats]) 
        if Preprocesser is None:
            ProcessedData, Preprocesser = self.minmax_scale(ProcessInput)
        else:
            ProcessedData = Preprocesser.transform(ProcessInput)

        return np.reshape(ProcessedData, [-1, Seqlen, Numfeats]), Preprocesser    

    def split_valid(self, Subjects):
        
        n = int(self.config['ValidationSplit'])
        
        num_valid = len(Subjects)//n
        SubjectsSet = Subjects.copy()
        random.Random(1).shuffle(SubjectsSet)

        ValidSet = SubjectsSet[0:num_valid]
        TrainSet = [Subject for Subject in Subjects if Subject not in ValidSet]

        return TrainSet, ValidSet

    def one_subject_out_cross_validation(self, Subjects, TestSubject):
        
        config = self.config
        model = Model(config, Subject = TestSubject)
        TrainSubjects, ValidSubjects = self.split_valid(Subjects)

        TrainData, TrainLabels = self.DataLoader.load_training_data(TrainSubjects, 'EEG', 'EMG', 'EEG_ACC')
        ValidData, ValidLabels = self.DataLoader.load_training_data(ValidSubjects, 'EEG', 'EMG', 'EEG_ACC')

        TrainEEG = TrainData['EEG']
        TrainACC = TrainData['EEG_ACC']
        TrainEMG = TrainData['EMG']

        ValidEEG = ValidData['EEG']
        ValidACC = ValidData['EEG_ACC']
        ValidEMG = ValidData['EMG']

        TrainEEG, PreprocesserEEG = self.standardize(TrainEEG)
        TrainACC, PreprocesserACC = self.minmax(TrainACC)
        TrainEMG, PreprocesserEMG = self.standardize(TrainEMG)

        ValidEEG, _ = self.standardize(ValidEEG, PreprocesserEEG)
        ValidACC, _ = self.minmax(ValidACC, PreprocesserACC)
        ValidEMG, _ = self.standardize(ValidEMG, PreprocesserEMG)

        self.save_preprocesser(PreprocesserEEG, TestSubject,'EEG')
        self.save_preprocesser(PreprocesserACC, TestSubject,'ACC')

        TrainDataInput = MultiModalDataLoading(TrainEEG, TrainACC, TrainEMG, TrainLabels)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataInput,  batch_size = self.batch_size, shuffle = True, worker_init_fn = self._set_seed(2048), generator = self._get_generator(2048))
        
        ValidData = MultiModalDataLoading(ValidEEG, ValidACC, ValidEMG, ValidLabels)
        ValidDataloader = torch.utils.data.DataLoader(ValidData,  batch_size = self.batch_size, shuffle = True, worker_init_fn = self._set_seed(2048), generator = self._get_generator(2048))
        
        model.fit(TrainDataloader, ValidDataloader)
        
        del TrainEEG, TrainEMG, TrainData, TrainDataloader, model
        del ValidEEG, ValidEMG, ValidData, ValidDataloader

        TestData = self.DataLoader.load_subject_data(TestSubject,'EEG','EEG_ACC')
        self.predictprobs(TestSubject, TestSubject, TestData, PreprocesserEEG, PreprocesserACC)
        locals().clear()

    def evaluating(self, Subjects, TestSubjects):        

        TestSubject = TestSubjects[0]
        model = Model(self.config, Subject = TestSubject)
        TrainSubjects, ValidSubjects = self.split_valid(Subjects)

        TrainData, TrainLabels = self.DataLoader.load_training_data(TrainSubjects, 'EEG', 'EMG', 'EEG_ACC')
        ValidData, ValidLabels = self.DataLoader.load_training_data(ValidSubjects, 'EEG', 'EMG', 'EEG_ACC')

        TrainEEG = TrainData['EEG']
        TrainACC = TrainData['EEG_ACC']
        TrainEMG = TrainData['EMG']

        ValidEEG = ValidData['EEG']
        ValidACC = ValidData['EEG_ACC']
        ValidEMG = ValidData['EMG']

        TrainEEG, PreprocesserEEG = self.standardize(TrainEEG)
        TrainACC, PreprocesserACC = self.minmax(TrainACC)
        TrainEMG, PreprocesserEMG = self.standardize(TrainEMG)

        ValidEEG, _ = self.standardize(ValidEEG, PreprocesserEEG)
        ValidACC, _ = self.minmax(ValidACC, PreprocesserACC)
        ValidEMG, _ = self.standardize(ValidEMG, PreprocesserEMG)

        self.save_preprocesser(PreprocesserEEG, TestSubject,'EEG')
        self.save_preprocesser(PreprocesserACC, TestSubject,'ACC')

        TrainDataInput = MultiModalDataLoading(TrainEEG, TrainACC, TrainEMG, TrainLabels)
        TrainDataloader = torch.utils.data.DataLoader(TrainDataInput,  batch_size = self.batch_size, shuffle = True, worker_init_fn = self._set_seed(2048), generator = self._get_generator(2048))
        
        ValidData = MultiModalDataLoading(ValidEEG, ValidACC, ValidEMG, ValidLabels)
        ValidDataloader = torch.utils.data.DataLoader(ValidData,  batch_size = self.batch_size, shuffle = True, worker_init_fn = self._set_seed(2048), generator = self._get_generator(2048))
        
        model.fit(TrainDataloader, ValidDataloader)
        
        del TrainEEG, TrainEMG, TrainData, TrainDataloader, model
        del ValidEEG, ValidEMG, ValidData, ValidDataloader
        
        for Subject in TestSubjects:
            TestData = self.DataLoader.load_subject_data(Subject,'EEG','EEG_ACC')
            self.predictprobs(TestSubject, Subject, TestData, PreprocesserEEG, PreprocesserACC)                      

    def predictprobs(self, ClassifierSubject, TestSubject, TestData, PreprocesserEEG, PreprocesserACC, TestEMG = None, TestLabels = None):
        
        networks = self.load_networks(ClassifierSubject)  
        
        networks.train(False)
        DataEEG = TestData['EEG']
        DataACC = TestData['EEG_ACC']
        # TestData = self.normalization(TestData)
        for rec in DataEEG.keys():
            iTestEEG = DataEEG[rec]
            iTestACC = DataACC[rec]
            iTestEEG, _ = self.standardize(iTestEEG, PreprocesserEEG)
            iTestACC, _ = self.minmax(iTestACC, PreprocesserACC)
            data = SingalModalDataLoading(iTestEEG, iTestACC)   
            dataloader = torch.utils.data.DataLoader(data,  batch_size = self.config['batch_size'], shuffle = False)    

            for i, (x1, x2) in enumerate(dataloader):
                x1 = x1.to(torch.float32)
                x2 = x2.to(torch.float32)
                if i == 0:
                    iprobs = networks.predict(x1, x2)
                    probs = iprobs.clone()
                else:
                    iprobs = networks.predict(x1, x2)
                    probs = torch.cat((probs, iprobs))
                
            self.save_predictpros(TestSubject, rec, probs)  
              
        locals().clear()

    def load_networks(self, test_subject):
        """
        Load the trained networks for the given subject.
        """
        model_path = os.path.join(self.classifier_path, test_subject, f'classifier-{test_subject}.pth')
        state_dict = torch.load(model_path)
        networks = Translator(self.config)
        networks.load_state_dict(state_dict)
        return networks

    def save_predict_probs(self, subject, recording, predict_probs):
        """
        Save the predicted probabilities for the given subject and recording.
        """
        subject_path = os.path.join(self.results_path, subject)
        os.makedirs(subject_path, exist_ok=True)
        file_suffix = '_Results_SD.pt' if self.config.get('SD', 2) == 1 else '_Results.pt'
        file_path = os.path.join(subject_path, f'{recording}{file_suffix}')
        torch.save(predict_probs, file_path)

    def save_preprocessor(self, preprocessor, subject, modality=None):
        """
        Save the preprocessor for the given subject and modality.
        """
        subject_path = os.path.join(self.save_root, subject)
        os.makedirs(subject_path, exist_ok=True)
        model_path = os.path.join(subject_path, f'Preprocessor{modality}-{subject}.pkl')
        joblib.dump(preprocessor, model_path)


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

    multi_modal = MultiModal(config)

    for test_subject in subjects_list:
        train_subjects = [subject for subject in subjects_list if subject != test_subject]
        print(f"Processing {test_subject}...")
        multi_modal.one_subject_out_cross_validation(train_subjects, test_subject)

    multi_modal.evaluating(subjects_list, test_list)
