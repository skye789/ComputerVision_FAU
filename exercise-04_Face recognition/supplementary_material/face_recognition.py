import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        # face = face - (131.0912, 103.8827, 91.4953)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912,
                                                        103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.reshape(reshaped, (1, 3, 224, 224))

        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
# 人脸鉴别又分为开放的(open)和封闭的(closed)两种，
# 后者假设输入的人脸照片一定属于预先定义的人群中的一个；而前者有可能输入的是任何人的照片。
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=1.05, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    # takes a new aligned face of known class from a video,
    # extracts its embedding, and stores it as a training sample in the gallery.
    def update(self, face, label):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)
        self.labels.append(label)

    # ToDo
    # Implement the predict method that assigns a class label to an aligned face using k-NN.
    def predict(self, face):
        # Closed-Set Protocol
        embedding = self.facenet.predict(face)
        X = np.append(self.embeddings, [embedding], axis=0)

        nbrs = NearestNeighbors(n_neighbors=self.num_neighbours + 1, algorithm='brute').fit(X)
        distances, indices = nbrs.kneighbors(X)

        idx_to_prediction = indices[-1, 1:]

        # posterior probability
        label_to_prediction = np.array(self.labels)[idx_to_prediction]
        label_counts = Counter(label_to_prediction)

        predicted_label = label_counts.most_common(1)[0][0]
        num_predicted_label = label_counts.most_common(1)[0][1]

        indices_distance = np.where(label_to_prediction == predicted_label)
        distances_label = np.squeeze(distances[-1, indices_distance])
        dist_to_prediction = distances_label[1]  # clostest distance of embedding

        prob = num_predicted_label / self.num_neighbours

        # Open-Set Protocol
        if dist_to_prediction > self.max_distance or prob < self.min_prob:
            predicted_label = "unknown"

        return predicted_label, prob, dist_to_prediction


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=5, max_iter=25):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo
    # extracts and stores an embedding for a new face
    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)

    # ToDo
    # Implement the k-means algorithm
    # Store the estimated cluster centers and the labels assigned to the faces.
    def fit(self):
        # kmean = KMeans(n_clusters=self.num_clusters, max_iter=self.max_iter, init='random')
        # kmean.fit(self.embeddings)
        # self.cluster_center = kmean.cluster_centers_
        # self.cluster_membership = kmean.labels_.tolist()  # Labels of each frame/picture
        converge = False
        iter = 0
        init_idx = random.sample(range(len(self.embeddings)), self.num_clusters)
        center = self.embeddings[init_idx, :]
        while (not converge) and (iter < self.max_iter):
            # calculate cluster_membership
            cluster_membership = np.empty_like(self.embeddings)
            for i in range(len(self.embeddings)):
                repeat_embedding = np.repeat([self.embeddings[i]],
                                             repeats=self.num_clusters,
                                             axis=0)
                dist_to_centers = np.linalg.norm(repeat_embedding - center, axis=1)  # num_center
                center_idx = np.argmin(dist_to_centers)
                cluster_membership[i] = center[center_idx]
            self.cluster_membership = cluster_membership

            # update cluster_center
            new_center = np.empty_like(center)
            for j in range(len(center)):
                idx_same_center = cluster_membership == center[j]
                new_center[j] = np.mean(self.embeddings[idx_same_center], axis=0)
            self.cluster_center = new_center

            # check converge or not
            iter+=1
            # converge = np.all(new_center == center)
            difference = np.sum(new_center - center)
            if difference < 1e4:
                converge = True
            center = new_center

    # ToDo
    # Once the clustering is done, we can re-identify a face by finding its best matching cluster.
    def predict(self, face):
        embedding = self.facenet.predict(face)
        N_repeat_embedding = np.repeat([embedding],
                                       repeats=self.num_clusters,
                                       axis=0)
        distances_to_clusters = np.linalg.norm(self.cluster_center - N_repeat_embedding, axis=1)
        predicted_label_idx = np.argmin(distances_to_clusters)
        # label_dict = ['Alan_Ball', 'Manuel_Pellegrini','Marina_Silva','Nancy_Sinatra','Peter_Gilmour']
        return predicted_label_idx, distances_to_clusters

# # ToDo
#  def predict(self, face):
#      temp = np.zeros((1,self.facenet.get_embedding_dimensionality()))
#      temp[0] = self.facenet.predict(face)
#      bf = cv2.BFMatcher()
#      votePool = {}
#      matches = bf.knnMatch(temp.astype(np.float32),self.cluster_center.astype(np.float32),k=self.num_clusters)
#
#      dis = []
#      label = []
#      for m in matches[0]:
#          dis.append(m.distance)
#          label.append(self.cluster_membership[m.trainIdx])
#      idx = np.where(dis==np.min(dis))[0]
#      print(idx)
#      return self.cluster_center[idx], np.min(dis)
