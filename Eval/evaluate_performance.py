# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:02:58 2021

@author: AD
"""

import numpy as np

import numpy as np

def evaluation(predictions: np.ndarray, labels: np.ndarray, ground_truth: np.ndarray, scale: int) -> dict:
    """
    Evaluate the performance of the anomaly detection model.
    
    Args:
        predictions (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels for the test data.
        ground_truth (np.ndarray): Ground truth labels.
        scale (int): Scale factor for time adjustment.
    
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    starts = []
    stops = []
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    delays = []
    true_positive_times = []
    false_positive_starts = []
    false_positive_stops = []
    detection_starts = []
    detection_stops = []
    
    adjusted_predictions = np.zeros(predictions.shape)
    
    for p in range(len(predictions)):
        if predictions[p] == 1 and predictions[p - 1] == 0:
            adjusted_predictions[p] = 1
    
    for i in range(1, len(labels)):
        if labels[i - 1] == 0 and labels[i] == 1:
            starts.append(i)
        elif labels[i - 1] == 1 and (labels[i] == 0 or i == len(labels) - 1):
            stops.append(i)
    
    num_seizures = len(starts)
    
    if num_seizures > 0:
        for i in range(num_seizures):
            start = starts[i]
            stop = stops[i]
            
            if sum(adjusted_predictions[start:stop]) > 0:
                true_positives += 1
                first_detection = np.where(adjusted_predictions[start:stop] == 1)[0][0]
                delays.append(int(first_detection * scale))
                true_positive_times.append(int((start + first_detection) * scale))
                
                for k in range(len(adjusted_predictions[start:stop])):
                    if adjusted_predictions[start + k] == 1 and adjusted_predictions[start + k - 1] == 0:
                        detection_starts.append(int((start + k) * scale))
                    if adjusted_predictions[start + k] == 0 and adjusted_predictions[start + k - 1] == 1:
                        detection_stops.append(int((start + k) * scale))
            else:
                false_negatives += 1
    
    if (num_seizures - true_positives) != false_negatives:
        print('Incorrect detection!')
        return
    
    predictions[ground_truth == 1] = 0
    
    for i in range(len(predictions)):
        if i == 0:
            if predictions[i] == 1:
                false_positives += 1
                false_positive_starts.append(int(i * scale))
        else:
            if predictions[i] == 1 and predictions[i - 1] == 0:
                false_positives += 1
                false_positive_starts.append(int(i * scale))
                detection_starts.append(int(i * scale))
            if i < len(predictions) - 1:
                if predictions[i] == 1 and predictions[i + 1] == 0:
                    false_positive_stops.append(int(i * scale))
                    detection_stops.append(int(i * scale))
            elif i == len(predictions) - 1:
                if predictions[i] == 1:
                    false_positive_stops.append(int(i * scale))
                    detection_stops.append(int(i * scale))
    
    false_positive_times = [false_positive_starts, false_positive_stops]
    detections = [detection_starts, detection_stops]
    
    false_alarms = false_positives / (len(predictions) * scale) * 3600
    
    evaluation_results = {
        'TP': true_positives, 
        'FN': false_negatives, 
        'FP': false_positives, 
        'True Positive Time':true_positive_times, 
        'Delay': delays, 
        'False Positive Times': false_positive_times, 
        'Detections':detections
        }
    
    
    return evaluation_results