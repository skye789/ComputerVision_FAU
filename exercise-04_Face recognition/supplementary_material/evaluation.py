import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='bytes')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='bytes')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, self.similarity = \
            self.classifier.predict_labels_and_similarities(self.test_embeddings)

        self.similarity_known = self.similarity[self.test_labels != UNKNOWN_LABEL]
        self.prediction_label_known = prediction_labels[self.test_labels != UNKNOWN_LABEL]

        # self.similarity_thresholds = []
        self.similarity_thresholds = self.select_similarity_threshold(self.similarity, self.false_alarm_rate_range)
        identification_rates = self.calc_identification_rate(self.prediction_label_known)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': self.similarity_thresholds,
                              'identification_rates': identification_rates}

        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        return np.percentile(similarity[self.test_labels==UNKNOWN_LABEL],(1-false_alarm_rate)*100)

    # takes a set of predicted class labels as input,
    # compares them to the target labels, and computes the identification rate at rank 1
    def calc_identification_rate(self, prediction_labels):
        iden_rates = []
        test_labels_known = self.test_labels[self.test_labels==UNKNOWN_LABEL]
        for t in self.similarity_thresholds:
            n_true = 0
            for pl,tl,s in zip(prediction_labels,test_labels_known,self.similarity_known):
                # print(s,t)
                if (tl == UNKNOWN_LABEL and s > t):
                    n_true += 1

                elif pl == tl:
                    n_true += 1
            acc = n_true/len(prediction_labels)
            iden_rates.append(acc)
        return iden_rates



