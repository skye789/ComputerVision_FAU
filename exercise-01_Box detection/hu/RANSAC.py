import random
import numpy as np
class RANSAC:
    def __init__(self, threshold= 0.02, max_iter_time=500):
        super().__init__()
        self.N = max_iter_time
        self.threshold = threshold
        self.data = None

    def loadPointCloud(self,pointCloud):
        # remove invalid measurements
        self.data = pointCloud[pointCloud[:,:,2]!=0]

    def fit(self):
        i = 0
        max_inliers = 0
        bestfit = None
        while (i<self.N):
            i += 1
            samples = self.get_samples()
            model = self.get_plane(samples)
            n_inliers = self.get_inliers(model)
            if(max_inliers<n_inliers):
                bestfit = model
                max_inliers = n_inliers
        return bestfit
    def get_inliers(self,model):
        mod_d = self.data[:,0]*model[0] + self.data[:,1]*model[1] + self.data[:,2]*model[2] + model[3]
        # mod_d = self.data.dot(model[:3])+model[3]
        mod_area = np.sqrt(np.sum(np.square([model[:3]])))
        d = abs(mod_d) / mod_area
        temp = np.zeros_like(d)
        temp[d<self.threshold] = 1
        return np.sum(temp)
    def get_plane(self,samples):
        p0,p1,p2 = samples
        v1 = p0-p1
        v2 = p1-p2
        n = np.cross(v1,v2)
        model = np.hstack((n,-(n[0]*p0[0]+n[1]*p0[1]+n[2]*p0[2])))
        return model
    def get_samples(self):
        dataSize = self.data.shape[0]
        sample_idx = random.sample(range(dataSize),3)
        p0 = self.data[sample_idx[0]]
        p1 = self.data[sample_idx[1]]
        p2 = self.data[sample_idx[2]]
        return p0,p1,p2




















