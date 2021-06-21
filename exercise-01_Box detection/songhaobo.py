import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from scipy.io import  loadmat
from scipy.ndimage import label, generate_binary_structure
import scipy.ndimage as nd
import cv2
import imutils
import math
from open3d import *
import random

def imagetoshow2D(img,colormap='gray'):
    plt.figure()
    plt.imshow(img,cmap=colormap)

    # plt.clf()

def imgetoshow3D(imgcloudflatten):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = imgcloudflatten[:, 0]
    y = imgcloudflatten[:, 1]
    z = imgcloudflatten[:, 2]
    ax.scatter(x, y, z,marker='.')
    plt.show()

def imgetoshow3DFast(imgcloudflatten):
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(imgcloudflatten)
    draw_geometries([point_cloud])

def trilinetest(N):
    summe=np.sum(N)
    if summe:
        return True
    else:
        return False

def simpleransac(cloudflatten,max_iterations, sigma):
    best_ic = 0
    best_model = None
    i=0
    inliersofwholecloud=0
    N=None
    while(i<max_iterations):
        point1 = cloudflatten[random.randint(0, cloudflatten.shape[0]-1), :]
        point2 = cloudflatten[random.randint(0, cloudflatten.shape[0]-1), :]
        point3 = cloudflatten[random.randint(0, cloudflatten.shape[0]-1), :]
        AB = np.asmatrix(point2 - point1)
        AC = np.asmatrix(point3 - point1)
        N = np.cross(AB, AC)  # 向量叉乘，求法向量
        # N = N/np.linalg.norm(N)
        if not trilinetest(N):
            print("3 point in a line")
            continue
        # Ax+By+Cz+D=0
        Ax = N[0, 0]
        By = N[0, 1]
        Cz = N[0, 2]
        D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])

        ic = 0
        normal_abc = math.sqrt((pow(Ax, 2) + pow(By, 2) + pow(Cz, 2)))
        for j in range(cloudflatten.shape[0]):
            mod_d = Ax * cloudflatten[j, 0] + By * cloudflatten[j, 1] + Cz * cloudflatten[j, 2] + D
            d = abs(mod_d)/normal_abc
            if d < sigma:
                ic += 1
        inliersofwholecloud = ic / cloudflatten.shape[0]
        try:
            if inliersofwholecloud == 1:
                raise ValueError("BAD threshold,maybe too big.All points are inlier ")
        except ValueError as e:
            print("Error：", repr(e))
        # print("inliersofwholecloud",inliersofwholecloud)
        max_iterations = 2 * math.log(1 - 0.999) / math.log(1.0 - pow(inliersofwholecloud, 3))
        i +=1
        if ic > best_ic:
            best_ic = ic
            best_model = [Ax, By, Cz, D]
            # if ic > min_inliers :
            #     break
    print('took iterations:', i , 'best model:', best_model,'Normal vector:',N,'Ratio:', inliersofwholecloud)
    try:
        if i == max_iterations:
            raise ValueError("NO fitted Model")
    except ValueError as e:
        print("Error：", repr(e))
    return np.array(best_model)

def maskgenerator(cloudflatten,img,best_model,sigmal):
    [a, b, c, d] = best_model
    mask = np.zeros_like(cloudflatten[:, 0])
    normal_abc = math.sqrt((pow(a, 2) + pow(c, 2) + pow(b, 2)))
    for index in range(cloudflatten.shape[0]):
        plane= a * cloudflatten[index, 0]+b * cloudflatten[index, 1]+c * cloudflatten[index, 2]+d
        distance = abs(plane)/normal_abc
        if distance < sigmal:
            mask[index] = 1
    point=cloudflatten[mask==1]
    pointonplane = point[random.randint(0, point.shape[0] - 1), :]

    maskreshape = np.reshape(mask, (img.shape[0], img.shape[1]))

    return maskreshape,pointonplane

def masksmooth(input):
    mask = nd.median_filter(input, 3)
    mask = nd.binary_closing(mask)
    mask = nd.binary_dilation(mask)
    return mask
def biggestcc(input):
    labeled_array, num_features = label(input)
    max_num = 0
    max_label = 0
    for i in range(1,
                   num_features + 1):  # todo not from 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 0 is background !!!!!!!!!!!!!!!!!!!!!!!
        if np.sum(labeled_array == i) > max_num:
            max_num = np.sum(labeled_array == i)
            max_label = i

    onlybox = (labeled_array == max_label)
    return onlybox

def distancepoint2plane(point,planemod):
    [Ax, By, Cz, D]=planemod

    normal_abc = math.sqrt((pow(Ax, 2) + pow(By, 2) + pow(Cz, 2)))
    mod_d = Ax * point[0] + By * point[1] + Cz * point[2] + D
    distance = abs(mod_d) / normal_abc
    return distance

def pointsored(edge_Coordinate, mask):
    min_x = np.min(edge_Coordinate[0]) #zuo
    max_x = np.max(edge_Coordinate[0])
    min_y = np.min(edge_Coordinate[1])
    max_y = np.max(edge_Coordinate[1])
    point0 = (min_x, np.min(np.where(mask[min_x] == 255)))#y,x-- col,row
    point1 = (max_x, np.min(np.where(mask[max_x] == 255)))
    point2 = (np.min(np.where(mask[:, min_y] == 255)), min_y)
    point3 = (np.min(np.where(mask[:, max_y] == 255)), max_y)
    return np.array([point0,point1 , point2,point3])

if __name__ == '__main__':
    filename = r"./files/example1kinect.mat"
    nr=list(filename)[-11]
    img = loadmat(filename)
    imgamp = img['amplitudes'+nr]
    imgdis = img['distances'+nr]
    imgcloud = img['cloud'+nr]
    imgcloudflatten = imgcloud.reshape((-1, 3))
    imgcloudflattenorigin = imgcloudflatten.copy()
    valid_index = np.nonzero(imgcloudflatten[:, 2])
    imgcloudflatten = np.squeeze(imgcloudflatten[valid_index, :], axis=0)

    thershold= 0.04
    max_iterations = 10000
    best_model=simpleransac(imgcloudflatten,1000,thershold)
    floormask,pointonfloorplane = maskgenerator(imgcloudflattenorigin,imgamp,best_model,thershold)
    floormask=masksmooth(floormask)*1
    maskinvert=1-floormask
    onlybox=biggestcc(maskinvert)
    boxcloud=imgcloud[onlybox==1]
    wherezeroVec = np.nonzero(boxcloud[:, 2])
    boxcloud = np.squeeze(boxcloud[wherezeroVec, :], axis=0)
    # imgetoshow3DFast(newbox)
    # imgetoshow3DFast(boxcloud)
 #----------------------------------------Find Box top plane-----------------------------------------------
    modelbox=simpleransac(boxcloud,max_iterations = 1000,sigma = 0.008)
    topmask,pointontopplane = maskgenerator(imgcloudflattenorigin,imgamp,modelbox,sigmal=0.008)
    topmask=masksmooth(topmask)*1
    topmask=biggestcc(topmask)
    topmask=(topmask*255).astype(np.uint8)
    blank = np.empty(topmask.shape, dtype=np.uint8)
    cnts = cv2.findContours(topmask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(blank, cnts, -1, (255, 255, 255), 1)
    pointcoordinate=np.where(blank==255)
    box=pointsored(pointcoordinate,blank)



    #
    # topmask = np.float32(topmask)
    # dst = cv2.cornerHarris(topmask, 2, 3, 0.04)
    #
    # dst_norm = np.empty(dst.shape, dtype=np.float32)
    # dst_norm2 = np.empty(dst.shape, dtype=np.float32)
    # cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # print(dst_norm.shape)
    # plt.figure()
    # plt.imshow(topmask)
    # for i in range(dst_norm.shape[0]):
    #     for j in range(dst_norm.shape[1]):
    #         if int(dst_norm[i, j]) > 95:
    #             # cv2.circle(dst_norm2, (j, i), 2, (255, 255, 255), 2)
    #             plt.plot(j,i, 'om')  # 绘制紫红色的圆形的点


    # sor=pointsored(pointcoordinate,blank)
    imagetoshow = np.zeros(topmask.shape)
    pointcoordinatetop=np.where(topmask==255)
    pointcoordinatefloor=np.where((1-onlybox)==1)
    imagetoshow[pointcoordinatetop]=1
    imagetoshow[pointcoordinatefloor]=2



    distance = distancepoint2plane(pointontopplane,best_model)
    print("distance:",distance)
    imagetoshow2D(imagetoshow)
    for index in range(len(box)):
        plt.plot(box[index][1], box[index][0], 'om')  # 绘制紫红色的圆形的点
    plt.show()
















