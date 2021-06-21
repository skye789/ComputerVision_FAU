import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
import math
from scipy.ndimage import filters
from pylab import *
import open3d
from skimage.feature import corner_harris, corner_subpix, corner_peaks

# def corner(mask_conn):
#     coords = corner_peaks(corner_harris(mask_conn), min_distance=16, threshold_rel=0.02)
#     coords_subpix = corner_subpix(mask_conn, coords, window_size=13)
#     fig, ax = plt.subplots()
#     ax.imshow(mask_conn, cmap=plt.cm.gray)
#     ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
#             linestyle='None', markersize=6)
#     ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
#     plt.show()

# def harris(img):
#     img= img*1
#     img = img*255
#     # img=img.astype(np.uint8)
#     # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#     #
#     # img = np.uint8(img)
#     #图像转换为float32
#     img = np.float32(img)
#     dst = cv2.cornerHarris(img, 2, 3, 0.04)
#     #result is dilated for marking the corners, not important
#     dst = cv2.dilate(dst, None)  # 图像膨胀
#     # Threshold for an optimal value, it may vary depending on the image.
#     # print(dst)
#     # img[dst>0.00000001*dst.max()]=[0,0,255] #可以试试这个参数，角点被标记的多余了一些
#     img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 角点位置用红色标记
#     # 这里的打分值以大于0.01×dst中最大值为边界
#
#     cv2.imshow('dst', img)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()


def imgetoshow3DFast(cloud):
    point_cloud = open3d.PointCloud()
    point_cloud.points = open3d.Vector3dVector(cloud)
    open3d.draw_geometries([point_cloud])

def get_valid_cloud(cloud):
    cloud = cloud.reshape((-1, 3))
    valid_index = np.nonzero(cloud[:, 2])
    valid_cloud = np.squeeze(cloud[valid_index, :], axis=0)
    return valid_cloud

def points_to_plane_distances(points, normal, plane_point):
    vectors = plane_point - points
    distances = abs(np.dot(normal, np.transpose(vectors)) / np.linalg.norm(normal))
    return distances

def model_solver(sample_points):
    # three points construct plane model (normal)
    vec1 = sample_points[0] - sample_points[1]
    vec2 = sample_points[0] - sample_points[2]
    normal = np.cross(vec1, vec2)
    plane_point = sample_points[0]
    return normal, plane_point


#RANSAC  use cloud to detect plane with the most inliers
def RANSAC_get_plane(cloud, threshold, k):
    i = 0
    max_inliers = 0
    P = 0.999   # accucacy of getting the right model
    model_solver_point = 3

    while(i < k):
        # randomly choose three points
        random_idx = np.random.choice(cloud.shape[0], size=model_solver_point)
        sample_points = cloud[random_idx, :]

        normal, plane_point = model_solver(sample_points)
        if(np.linalg.norm(normal)==0): continue
        distances = points_to_plane_distances(cloud, normal, plane_point)
        inlier_number = np.count_nonzero(distances < threshold)
        inlier = cloud[distances < threshold]
        outlier = cloud[distances >= threshold]

        # estimate whether model gets better or not
        if inlier_number > max_inliers:
            w = inlier_number/cloud.shape[0]
            k = 2 * math.log(1 - P) / math.log(1 - pow(w, 3))
            # k = 100;
            max_inliers = inlier_number
            best_model_normal = normal
            best_model_plane_point = plane_point

            # print(w)
        i += 1
    # print('iter num = ', i)
    return best_model_normal, best_model_plane_point, inlier, outlier

def get_plane_mask(cloud, valid_cloud, threshold):
    # maximum iteration number, update in each while-process
    valid_cloud = valid_cloud[valid_cloud[:, 2]!=0]
    mask_shape = cloud.shape[0:2]
    cloud = cloud.reshape((-1, 3))
    k = 1000
    best_model_normal, best_model_plane_point, inlier, outlier = RANSAC_get_plane(valid_cloud, threshold, k)
    distances = points_to_plane_distances(cloud, best_model_normal, best_model_plane_point)
    mask = np.where(distances < threshold, 1, 0).reshape(mask_shape)
    # mask = (distances < threshold).reshape(mask_shape)
    return mask, best_model_normal, best_model_plane_point

#find the maximum connected region
def filter_mask(mask, order):
    erosion = scipy.ndimage.morphology.binary_erosion(mask)
    label, num_features = ndimage.label(erosion)
    region_sizes = ndimage.sum(erosion, label, range(num_features + 1))
    sort_index = np.argsort(region_sizes)
    sort_index = sort_index[::-1]   # from max to min
    filtered_mask = np.where(label == sort_index[order], 1, 0)
    # filtered_mask = (label == sort_index[order])
    return filtered_mask

def get_margin(mask):
    dx = ndimage.sobel(mask, axis=0)
    dy = ndimage.sobel(mask, axis=1)
    margin = np.hypot(dx, dy)
    margin = np.where(margin>0, 1, 0)
    return margin

def get_height(normal, p_one, p_two):
    vec = p_one - p_two
    height = abs(np.dot(vec, normal))/np.linalg.norm(normal)
    return height

def get_corner(mask):
    cood = np.where(mask==1)
    min_x = np.min(cood[0])
    max_x = np.max(cood[0])
    min_y = np.min(cood[1])
    max_y = np.max(cood[1])
    point0 = ( np.min(np.where(mask[min_x, :] == 1)), min_x)
    point1 = ( np.min(np.where(mask[max_x, :] == 1)), max_x)
    point2 = ( min_y, np.min(np.where(mask[:, min_y] == 1)))
    point3 = (max_y, np.min(np.where(mask[:, max_y] == 1)))
    return point0, point1, point2, point3

if __name__ == "__main__":
    # read data
    dataFile = 'data/example3kinect.mat'
    data = scio.loadmat(dataFile)
    cloud = data['cloud3']
    amplitude = data['amplitudes3']
    # distance = data['distances2']

    valid_cloud = get_valid_cloud(cloud)

    floor_mask, floor_n, floor_p = get_plane_mask(cloud, valid_cloud, 0.05)

    filter_floor_mask = filter_mask(floor_mask, 0)
    box_cloud = cloud[filter_floor_mask == 0] #box is second largest connected region
    top_mask, top_n, top_p = get_plane_mask(cloud, box_cloud, 0.02)
    top_mask = filter_mask(top_mask, 0)
    # top_filtered_mask = filter_mask(top_mask, 0)
    top_margin = get_margin(top_mask)

    height1 = get_height(floor_n, floor_p, top_p)
    height2 = get_height(top_n, floor_p, top_p)
    height = (height1 + height2) / 2
    print("height is ", height)

    cor_left, cor_right, cor_up, cor_down = get_corner(top_margin)
    length = np.linalg.norm(cloud[cor_left]-cloud[cor_up])
    print("length is ", length)
    width = np.linalg.norm(cloud[cor_left]-cloud[cor_down])
    print("width is ", width)
    volume = height* length *width
    print("volume is ", volume)

    # imgetoshow3DFast(cloud.reshape(-1,3))
    # imgetoshow3DFast(box_cloud)
    plt.subplot(231)
    plt.imshow(amplitude)
    plt.subplot(232)
    plt.imshow(floor_mask)
    plt.subplot(233)
    plt.imshow(filter_floor_mask)
    plt.subplot(234)
    plt.imshow(top_mask)
    plt.subplot(235)
    plt.imshow(top_margin)
    plt.subplot(236)
    plt.imshow(amplitude)
    plt.scatter(*cor_left, s=25, c='r')  # stroke, colour
    plt.scatter(*cor_right, s=25, c='r')  # stroke, colour
    plt.scatter(*cor_up, s=25, c='r')  # stroke, colour
    plt.scatter(*cor_down, s=25, c='r')  # stroke, colour
    plt.plot([cor_left[0],cor_up[0]], [cor_left[1],cor_up[1]], c='r')
    plt.plot([cor_right[0], cor_down[0]], [cor_right[1], cor_down[1]], c='r')
    plt.plot([cor_left[0], cor_down[0]], [cor_left[1], cor_down[1]], c='r')
    plt.plot([cor_right[0], cor_up[0]], [cor_right[1], cor_up[1]], c='r')
    plt.show()















