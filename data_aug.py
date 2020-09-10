import os
import random
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import multiprocessing

import warnings
warnings.filterwarnings('ignore')


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Function to distort image, based on https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    # random_state.rand(a,b) will generate a rand array with size (a, b) and range(0,1)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
         center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    if 1:
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def img_distortion(im, im_mask):
    # img : (0,255)  imgmask(0,1)
    imshape = im.shape
    im_maskshape = im_mask.shape
    if len(imshape) == 2:
        im = im[..., np.newaxis]
        imshape = im.shape
    if len(im_maskshape) == 2:
        im_mask = im_mask[..., np.newaxis]
    im_merge = np.concatenate((im, im_mask), axis=2)
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 10, im_merge.shape[1] * 0.06,
                                        im_merge.shape[1] * 0.06)
    imgout = im_merge_t[..., :imshape[2]].astype('uint8')
    maskout = im_merge_t[..., imshape[2]:]

    return imgout, maskout


def main(fn):

    dir_image = 'data/train/images_sub'
    dir_label = 'data/train/labels_sub'

    label = Image.open(os.path.join(dir_label, fn))
    image = Image.open(os.path.join(dir_image, fn))

    label_arr = np.array(label)
    image_arr = np.array(image)

    for i in range(3):

        image_dis, label_dis = img_distortion(image_arr, label_arr)

        label_dis = np.squeeze(label_dis)

        image_dis = Image.fromarray(image_dis.astype(np.uint8))
        label_dis = Image.fromarray(label_dis.astype(np.uint8))

        image_dis.save(os.path.join(dir_image, fn.replace('.png', '_dis%d.png' % (i+1))))
        label_dis.save(os.path.join(dir_label, fn.replace('.png', '_dis%d.png' % (i+1))))


if __name__ == '__main__':

    # # distort image and label
    # for fn in crack_list:
    #     main(fn)

    # # remove
    # for fn in bg_list_delete:
    #     os.remove(os.path.join(dir_image, fn))
    #     os.remove(os.path.join(dir_label, fn))

    dir_image = 'data/train/images_sub'
    dir_label = 'data/train/labels_sub'

    crack_list = []
    bg_list = []

    img_list = os.listdir(dir_image)
    label_list = os.listdir(dir_label)

    for fn in label_list:
        label = Image.open(os.path.join(dir_label, fn))
        label_arr = np.array(label)
        if label_arr.sum() > 256:
            crack_list.append(fn)
        else:
            bg_list.append(fn)

    bg_list_delete = random.choices(bg_list, k=int(0.7 * len(bg_list)))

    print(len(crack_list))
    print(len(bg_list))
    print(len(bg_list_delete))

    # to avoid multiple aug
    crack_list = [x for x in crack_list if 'dis' not in crack_list]

    pool = multiprocessing.Pool(processes=8)
    pool.map(main, crack_list)
    pool.close()
    pool.join()




