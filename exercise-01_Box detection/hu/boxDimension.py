import numpy as np
class BoxDimensionEstimator:
    def __init__(self,pointCloud,model_floor,mask_top):
        self.cloud = pointCloud
        self.model = model_floor
        self.mask = mask_top
        self.corner = None

    def get_volume(self):
        self.corner = self.compute_cornerCood()

        p0,p1,p2,p3 = self.corner

        edge1 = (self.get_distance(p0,p1) + self.get_distance(p2,p3))/2
        edge2 = (self.get_distance(p0,p3) + self.get_distance(p1,p2))/2

        h = self.get_height()
        return edge1*edge2*h

    def get_height(self):
        m = self.model
        imgCood = np.mean(self.corner,axis=1).astype(np.int)
        p = self.cloud[imgCood[0],imgCood[1]]
        h = m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3]
        h = abs(h)/np.sqrt(np.sum(np.square([m[:3]])))
        print("height is ", h)
        return h

    def compute_cornerCood(self):
        cood = np.where(self.mask==1)
        min_x = np.min(cood[0])
        max_x = np.max(cood[0])
        min_y = np.min(cood[1])
        max_y = np.max(cood[1])
        point0 = (min_x,np.min(np.where(self.mask[min_x]==1)))
        point1 = (max_x,np.min(np.where(self.mask[max_x]==1)))
        point2 = (np.min(np.where(self.mask[:,min_y]==1)),min_y)
        point3 = (np.min(np.where(self.mask[:,max_y]==1)),max_y)
        return point0,point2,point1,point3

    def get_distance(self,p1,p2):
        worldCood1 = self.cloud[p1[0],p1[1]]
        worldCood2 = self.cloud[p2[0],p2[1]]
        return np.sqrt(np.sum((worldCood1-worldCood2)**2))

    def get_corner(self):
        return self.corner

