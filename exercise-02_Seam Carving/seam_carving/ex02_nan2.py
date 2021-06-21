import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import minimum_filter1d,median_filter
from scipy.signal import find_peaks
from scipy.ndimage import binary_erosion
'''''
 1:remove kestrel
 2:mask remove bird
 
 3:seam carving crop bird
 4:seam carving crop vincent

 5:extend vincent
 
 6:text_line_segmentation
'''''
test_case = 6

crop_scale_c = 0.5  ##图片缩减20％
extend_scale_c = 0.8  ##图片扩大180％
path_std = 100  ##部分更新cost

kernal = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

def energy_function(img):
    con_0 = np.abs(signal.convolve2d(img[:, :, 0], kernal, mode='same'))
    con_1 = np.abs(signal.convolve2d(img[:, :, 1], kernal, mode='same'))
    con_2 = np.abs(signal.convolve2d(img[:, :, 2], kernal, mode='same'))
    return con_0 + con_1 + con_2

def cum_cost(cost):
    cum = np.ones_like(cost)
    cum[0, :] = cost[0, :]
    for i in range (1,cum.shape[0]):
        cum[i, :] = cost[i, :] + minimum_filter1d(cum[i - 1, :], size=3, mode='reflect')
    return cum

def removing(img):
    for _ in tqdm(range(10)):
        cost = energy_function(img)
        row_cost = np.sum(cost, axis=1)
        row_idx = np.argsort(row_cost)
        col_cost = np.sum(cost, axis=0)
        col_idx = np.argsort(col_cost)
        img = np.delete(img, row_idx[0:100], 0)
        img = np.delete(img, col_idx[0:150], 1)
    return img

def mask_remove(img, mask):
    for _ in tqdm(range(30)):
        cost = np.add(energy_function(img), mask)
        row_cost = np.sum(cost, axis=1)
        row_idx = np.argsort(row_cost)
        col_cost = np.sum(cost, axis=0)
        col_idx = np.argsort(col_cost)
        img = np.delete(img, row_idx[0:10], 0)
        img = np.delete(img, col_idx[0:20], 1)
        mask = np.delete(mask, row_idx[0:10], 0)
        mask = np.delete(mask, col_idx[0:20], 1)
    return img

def path(arr):
    rows, cols = arr.shape
    path_idx = np.zeros(rows)
    path_cols = np.zeros(rows)
    j = np.argmin(arr[-1, :])
    path_idx[-1] = (rows-1) * cols + j
    path_cols[-1] = j
    for i in reversed(range(0, rows-1)):
        if j == 0:
            j += np.argmin(arr[i, j : j+2])
        elif j == cols-1:
            j = j - (1 - np.argmin(arr[i, j-1: j+1]))
        else:
            j = j - (1 - np.argmin(arr[i, j-1: j + 2]))
        path_idx[i] = i * cols + j
        path_cols[i] = j

    path_mean = np.mean(path_cols)
    left = np.clip(np.round(path_mean - path_std).astype(int), 0, cols)
    right = np.clip(np.round(path_mean + path_std + 1).astype(int), 0, cols)
    return path_idx, left, right

def remove_path(arr, path_idx):
    if len(arr.shape) == 2:
        rows, cols = arr.shape
        arr = arr.reshape(rows * cols)
        new_arr = np.delete(arr, path_idx.astype(int), axis=0).reshape(rows, cols - 1)
    if len(arr.shape) == 3:
        rows, cols, deps = arr.shape
        arr = arr.reshape(rows * cols, deps)
        new_arr = np.delete(arr, path_idx.astype(int), axis=0).reshape(rows, cols - 1, deps)
    return new_arr

def seam_carving(img, cost, protect_mask=None):
    cum = cum_cost(cost)
    path_idx, left, right = path(cum)
    next_img = remove_path(img, path_idx)
    next_cost = remove_path(cost, path_idx)
    next_cost[:, left:right] = energy_function(next_img[:, left:right])

    next_protect_mask = protect_mask
    if protect_mask is not None:
        next_protect_mask = remove_path(protect_mask, path_idx)
        next_cost += next_protect_mask

    return next_img, next_cost, next_protect_mask


def add_path(arr, path_idx):
    path_idx = path_idx.astype(int)
    if len(arr.shape) == 2:
        rows, cols = arr.shape
        arr = arr.reshape(rows * cols)
        path = arr[path_idx]
        neibor = arr[path_idx + 1]
        new_path = (path + neibor)//2
        new_arr = np.insert(arr, path_idx+1, new_path, axis=0).reshape(rows, cols+1)
    if len(arr.shape) == 3:
        rows, cols, deps = arr.shape
        arr = arr.reshape(rows * cols, deps)
        path = arr[path_idx, :]
        neibor = arr[path_idx + 1, :]
        new_path = (path.astype(int) + neibor.astype(int)) // 2
        # new_path = np.zeros_like(path)
        new_arr = np.insert(arr, path_idx+1, new_path, axis=0).reshape(rows, cols+1, deps)
    return new_arr

def add_path_seam_mask(arr, path_idx):
    path_idx = path_idx.astype(int)
    rows, cols = arr.shape
    arr = arr.reshape(rows * cols)
    new_path = 255 * np.ones_like(path_idx)
    new_arr = np.insert(arr, path_idx+1, new_path, axis=0)
    # new_arr[path_idx-1] = new_path
    for i in range(-1, 1):
        new_arr[path_idx+i] += new_path
    new_arr = new_arr.reshape(rows, cols + 1)
    return new_arr

def upscaling(img, cost, seam_mask):
    cum = cum_cost(cost)
    path_idx, left, right = path(cum)
    next_img = add_path(img, path_idx)
    next_seam_mask = add_path_seam_mask(seam_mask, path_idx)
    next_cost = energy_function(next_img) + next_seam_mask
    return next_img, next_cost, next_seam_mask

def crop_or_extend(img, protect_mask=None, mode='crop'):
    rows, cols, _ = img.shape
    crop_iter_num = int(crop_scale_c * rows)
    extend_iter_num = int(extend_scale_c * rows)

    if protect_mask is not None:
        cost = energy_function(img) + protect_mask
    else:
        cost = energy_function(img)

    seam_mask = np.zeros_like(cost)

    if mode == 'crop':
        for _ in tqdm(range(crop_iter_num)):
            img, cost, protect_mask = seam_carving(img, cost, protect_mask)
    if mode == 'extend':
        for _ in tqdm(range(extend_iter_num)):
            img, cost, seam_mask = upscaling(img, cost, seam_mask)

    return img

def text_line_Seg(img):
    img_array = np.array(img)
    img_gray = np.array(img.convert('L'))
    rows, cols = img_gray.shape
    threshold = np.mean(img_gray)
    binary_img = np.where(img_gray<threshold, 0, 255)
    subImg = binary_img[:, :int(cols * 0.25)]

    projection = np.sum(subImg, axis=1)
    projection = median_filter(projection, size=20)
    peaks, _ = find_peaks(projection, distance=70)

    # plt.plot(projection)
    # plt.plot(peaks, projection[peaks], "x")
    # plt.show()

    # set cells in left border = m, except line start = 0
    cost = energy_function(img_array)
    # cost = np.abs(signal.convolve2d(img_gray, kernal, mode='same'))
    m = np.amax(np.sum(cost, axis=1))  ##m is row_sum_cost
    cost[:, 0] = m
    cost[peaks, 0] = 0

    # Compute the cumulative costs and the path
    cum = cum_cost(cost.T).T
    num_peaks = np.size(peaks)
    mask = np.zeros_like(cost, dtype=np.bool)

    rows, cols = cum.shape

    for n in range(num_peaks):
        i = peaks[n]

        mask[i, 0] = True
        for j in range(1, cols):
            if i == 0:
                i += np.argmin(cum[i: i + 2, j])
            elif i == rows - 1:
                i += np.argmin(cum[i - 1: i + 1, j]) - 1
            else:
                i += np.argmin(cum[i - 1: i + 2, j]) - 1
            mask[i, j] = True

    img_array[mask, :] = np.array([255, 0, 0])

    plt.imshow(img_array)
    plt.show()


if __name__=="__main__":

    # # test cum_cost and path
    # arr = np.array([[4, 3, 2, 1, 3, 5, 4], [2, 5, 4, 3, 5, 1, 3], [4, 1, 3, 2, 4, 4, 2], [1, 5, 3, 2, 5, 1, 1],
    #                    [4, 2, 1, 3, 2, 2, 4], [5, 2, 5, 5, 2, 4, 1], [3, 5, 1, 4, 1, 2, 5]])
    # cum = cum_cost(arr)
    # # print(cum)
    # idx_path = path(cum)
    # print(idx_path)

    if test_case == 1:
        img = Image.open('additional-data/common-kestrel.jpg')
        img_array = np.array(img)

        crop_output = removing(img_array)

        plt.subplot(121)
        plt.imshow(img)
        plt.title('kestrel')
        plt.subplot(122)
        plt.imshow(crop_output)
        plt.title('removing')
        plt.show()

    if test_case == 2:
        img = Image.open('additional-data/kingfishers.jpg')
        img_array = np.array(img)
        protect_mask = Image.open('additional-data/kingfishers-mask.png').convert('L')  # img.convert('L')为灰度图像
        protect_mask_array = np.array(protect_mask)

        crop_output = mask_remove(img_array, protect_mask_array)

        plt.subplot(121)
        plt.imshow(img)
        plt.title('kestrel')
        plt.subplot(122)
        plt.imshow(crop_output)
        plt.title('mask_remove')
        plt.show()

    if test_case == 3:
        img = Image.open('additional-data/kingfishers.jpg')
        img_array = np.array(img)
        protect_mask = Image.open('additional-data/kingfishers-mask.png').convert('L')  # img.convert('L')为灰度图像
        protect_mask_array = np.array(protect_mask)

        crop_output = crop_or_extend(img_array, protect_mask_array, mode='crop')

        plt.subplot(121)
        plt.imshow(img)
        plt.title('kestrel')
        plt.subplot(122)
        plt.imshow(crop_output)
        plt.title('crop')
        plt.show()

    if test_case == 4:
        img = Image.open('additional-data/vincent-on-cliff.jpg')
        img_array = np.array(img)
        protect_mask_array = None

        crop_output = crop_or_extend(img_array, protect_mask_array, mode='crop')

        plt.subplot(121)
        plt.imshow(img)
        plt.title('vincent')
        plt.subplot(122)
        plt.imshow(crop_output)
        plt.title('crop')
        plt.show()


    if test_case == 5:
        img = Image.open('additional-data/vincent-on-cliff.jpg')
        img_array = np.array(img)
        protect_mask_array = None

        crop_output = crop_or_extend(img_array, protect_mask_array, mode='extend')

        plt.subplot(211)
        plt.imshow(img)
        plt.title('vincent')
        plt.subplot(212)
        plt.imshow(crop_output)
        plt.title('extend')
        plt.show()

    if test_case == 6:
        img = Image.open('additional-data/e-codices_acv-P-Antitus_027v_large.jpg')
        text_line_Seg(img)










