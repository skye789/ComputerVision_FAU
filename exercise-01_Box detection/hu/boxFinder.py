from scipy.io import loadmat
from scipy.ndimage.morphology import binary_opening,binary_closing
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from RANSAC import RANSAC
from boxDimension import BoxDimensionEstimator

PATH = "files/"

THRESHOLD = 0.01
S_SIZE = 5
EXAMPLE_NUMMBER = 3
DEBUG_IMPORT = False
DEBUG_RANSAC = True
DEBUG_VISUAL = True

def get_mask(t,model,pointCloud,switchMask=False):
    mask = np.zeros_like(pointCloud[:,:,0])
    d = pointCloud[:,:,0]*model[0] + pointCloud[:,:,1]*model[1] + pointCloud[:,:,2]*model[2] + model[3]
    d = abs(d)/np.sqrt(np.sum(np.square([model[:3]])))
    mask[d<t] = 1
    if switchMask:
        mask = abs(mask-1)
    return mask

def visualization_mask(amp,mask_floor,mask_top,corner,ax):
    maskOnly = np.zeros_like(mask_floor)
    maskOnly += mask_floor
    maskOnly += mask_top*2
    temp = None
    for corner in corners:
        ax.scatter(corner[1],corner[0],c="r")
        if temp is not None:
            ax.plot([temp[1],corner[1]],[temp[0],corner[0]],c="r")
        temp = corner
    ax.plot([corners[0][1],corners[-1][1]],[corners[0][0],corners[-1][0]],c="r")
    ax.imshow(maskOnly)

def visualization_amp(amp,mask_floor,mask_top,corner,ax):
    temp = None
    for corner in corners:
        ax.scatter(corner[1],corner[0],c="r")
        if temp is not None:
            ax.plot([temp[1],corner[1]],[temp[0],corner[0]],c="r")
        temp = corner
    ax.plot([corners[0][1],corners[-1][1]],[corners[0][0],corners[-1][0]],c="r")
    ax.imshow(amp)

if __name__ == '__main__':
    # Load the data
    n_example = str(EXAMPLE_NUMMBER)
    importData = loadmat(PATH + "example" + n_example + "kinect.mat")

    # extract data from dict
    if n_example == "4":
        amplitude = importData["amplitudes"+ "3"]
    else:
        amplitude = importData["amplitudes"+ n_example]
    distances = importData["distances" + n_example]
    pointCloud = importData["cloud" + n_example]

    if DEBUG_IMPORT:
        # visualize the data
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2= fig.add_subplot(133, projection='3d')
        ax0.imshow(amplitude)
        ax1.imshow(distances)
        for i in range(pointCloud.shape[0]):
            for j in range(pointCloud.shape[1]):
                if (i%25==0) and (j%25==0):
                    ax2.scatter(pointCloud[i,j,0],pointCloud[i,j,1],pointCloud[i,j,2],c="b",marker=".")
        plt.show()

    # do RANSAC
    ransac = RANSAC(threshold=THRESHOLD)
    ransac.loadPointCloud(pointCloud)
    model = ransac.fit()
    mask = get_mask(THRESHOLD,model,pointCloud)
    if DEBUG_RANSAC:
        # visualize the mask
        fig2 = plt.figure()
        ax0 = fig2.add_subplot(231)
        ax1 = fig2.add_subplot(232)
        ax2 = fig2.add_subplot(233)
        ax3 = fig2.add_subplot(234)
        ax4 = fig2.add_subplot(235)
        ax5 = fig2.add_subplot(236)
        ax0.imshow(mask)

    # do opening
    mask = binary_closing(mask,structure=np.ones((S_SIZE,S_SIZE))).astype(np.int)
    if DEBUG_RANSAC:
        # visualize the mask
        ax1.imshow(mask)
        ax2.imshow(mask*amplitude)



    filteredPointCloud = np.zeros_like(pointCloud)
    mask_remove_floor = abs(mask-1)
    for i in range(3):
        filteredPointCloud[:,:,i] = pointCloud[:,:,i] * mask_remove_floor

    ransac = RANSAC(threshold=THRESHOLD)
    ransac.loadPointCloud(filteredPointCloud)
    model2 = ransac.fit()
    mask_top = get_mask(THRESHOLD,model2,filteredPointCloud)

    if DEBUG_RANSAC:
        # visualize the mask
        ax3.imshow(mask_top)

    mask_top = binary_closing(mask_top,structure=np.ones((S_SIZE,S_SIZE))).astype(np.int)

    # find the largest connected component
    conn = label(mask_top)
    h = np.histogram(conn[0],conn[1]+1)[0]
    h[h==np.max(h)] = 0
    idx = np.where(h==np.max(h))[0][0]
    mask_conn = np.zeros_like(mask)
    mask_conn[conn[0]==idx] = 1
    if DEBUG_RANSAC:
        # visualize the mask
        ax4.imshow(conn[0])
        ax5.imshow(mask_conn*amplitude)

    estimator = BoxDimensionEstimator(pointCloud,model,mask_conn)
    volume = estimator.get_volume()
    corners = estimator.get_corner()

    if DEBUG_VISUAL:
        fig3 = plt.figure("Visualization of the box")
        ax0 = fig3.add_subplot(121)
        ax0.set_title("visualization with masks")
        ax1 = fig3.add_subplot(122)
        ax1.set_title("visualization with amplitude")
        visualization_mask(amplitude,mask,mask_top,corners,ax0)
        visualization_amp(amplitude,mask,mask_top,corners,ax1)
    print("volume of the box is {:.2f} m^3".format(volume))

    plt.show()