'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
from skimage.color import rgb2hsv
import skimage.transform
import skimage.util
import skimage.segmentation as segmentation
import numpy as np


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    # mask对每一个像素都进行编号
    mask = segmentation.felzenszwalb(im_orig, scale, sigma, min_size)
    seg_img = np.zeros((im_orig.shape[0], im_orig.shape[1], 4))
    seg_img[:, :, :3] = im_orig
    seg_img[:, :, 3] = mask

    return seg_img


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    sum = 0
    for a, b in zip(r1["hist_color"], r2["hist_color"]):
        con = np.concatenate(([a], [b]), axis=0)
        sum += np.sum(np.min(con, axis=0))
    return sum

def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    sum = 0
    for a, b in zip(r1["hist_text"], r2["hist_text"]):
        con = np.concatenate(([a], [b]), axis=0)
        sum += np.sum(np.min(con, axis=0))
    return sum


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    return 1 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    # bb is Bounding Box
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    # input的img已flatten,输入的img参数是每一类别的所有像素点的hsv值

    BINS = 25
    hist = np.zeros((BINS, 3))
    for i in range(3):
        # hist[:, i], _ = np.histogram(img[:, i], bins=BINS, range=(0, 255))
        hist[:, i], _ = np.histogram(img[:, i], bins=BINS, range=(0, 255))
    # L1 normalize
    hist = hist / len(img)
    return hist


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    # LBP是一种可用于纹理分类的不变描述符
    ret = np.zeros_like(img)
    for i in range(3):
        ret[:, :, i] = skimage.feature.local_binary_pattern(img[:, :, i], P=8, R=1)

    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.zeros((BINS, 3))
    for i in range(3):
        hist[:, i], _ = np.histogram(img[:, i], bins=BINS, range=(0, 1))

    # L1 normalize
    hist = hist / len(img)
    return hist


# 提取区域的尺寸，颜色和纹理特征
def extract_regions(img):
    """
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    """
    # R记录了该图像每个region的信息：mix_x,min_y,max_x,max_y,size,hist_c,hist_t
    R = {}

    hsv_img = rgb2hsv(img[:, :, :3])

    # Count pixel positions
    # 遍历图片像素点,将每个region最大以及最小的x、y坐标记录在字典R中
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            label = img[y, x, 3]

            # initialize a new region
            if label not in R:
                R[label] = {"labels": label, "min_x": x, "max_x": x, "min_y": y, "max_y": y}
            # 改写R字典里的value
            if x > R[label]["max_x"]: R[label]["max_x"] = x
            if y > R[label]["max_y"]: R[label]["max_y"] = y
            if x < R[label]["min_x"]: R[label]["min_x"] = x
            if y < R[label]["min_y"]: R[label]["min_y"] = y

    texture_grad = calc_texture_gradient(img)
    for label in R.keys():
        mask = (img[:, :, 3] == label)
        R[label]["size"] = hsv_img[mask].shape[0]  # hsv_img[mask].shape=(51,3)
        # calculate color and texture histograms
        R[label]["hist_color"] = calc_colour_hist(hsv_img[mask])
        R[label]["hist_text"] = calc_texture_hist(texture_grad[mask])

    return R


# 找邻居 -- 通过计算每个区域与其余的所有区域是否有相交，来判断是不是邻居
def extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    R = list(regions.items()) # [("key", value)]
    for cur, a in enumerate(R[:-1]):  # enumerate()用于迭代，可返回下标和值，即cur是下标，从0开始,遍历R所有region(除最后一个)
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):  # 拿当前region-a与a之后的所有类别进行比较 # a[0]是key；a[1]是value (因为R转为list)
                neighbours.append((a, b))  # 将是邻居的两个region装进数组中
    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    ### YOUR CODE HERE
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_color": (r1["hist_color"] * r1["size"] + r2["hist_color"] * r2["size"]) / new_size,
        "hist_text": ( r1["hist_text"] * r1["size"] + r2["hist_text"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)  # R有每个region的信息

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)  # neighbours为所有是邻居的region的集合

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        # ai，bi指的是两个region的key(即label)，ar，br指的是两个region的value(即region信息)
        # S是一个字典，key是两个类别的编号，value是这两个类别的相似度
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:
        # Get highest similarity
        # 找到S中相似度最高的两个region的label--i,j
        # sorted函数对S进行排序，[-1][0]指的是取出排序后最后一项（value即相似度最大）的第一个元素（即两个region的label）
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0  # 新建一个新的区域编号
        R[t] = merge_regions(R[i], R[j])
        # 为什么不删除旧的region????, 后面cnn处理?


        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                #将S中有类别i和类别j的项目找出来放入key_to_delete数组中
                key_to_delete.append(k)

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for k in key_to_delete:
            del S[k]

        # task 5、6的例子: S的key里(1,3)最像，新区域为的label为5. key_to_delete包含(1,2)(1,3)(1,4)(2,3)(2,4)
        # task7要重算(2,5)(4,5)的相似度并放入S

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###
        for k in key_to_delete:
            if k != (i, j):
                x = k[0]
                if k[0] is i or k[0] is j:
                    x = k[1]
                S[(t, x)] = calc_sim(R[t], R[x], imsize)
   # while循环结束，已经合并结束，R更新完成

    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for k, r in list(R.items()):
        # print("r=",r)
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    return image, regions
