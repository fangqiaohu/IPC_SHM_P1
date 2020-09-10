


def main(i):

    import cv2
    import os
    import numpy as np
    import random

    full_h = 3264
    full_w = 4928
    resize_h = 3072
    resize_w = 4608
    sub_size = 512
    stride = 128
    overlap = sub_size - stride
    resize = True

    image_root_path = '../_raw_data/train/images'
    label_root_path = '../_raw_data/train/labels'

    image_list = os.listdir(image_root_path)
    label_list = os.listdir(label_root_path)

    # crack color
    lower_red = np.array([0, 0, 120], dtype="uint8")
    upper_red = np.array([0, 0, 140], dtype="uint8")

    # ######## Binary label ########
    # for i in range(len(image_list)):
    #     label = cv2.imread(os.path.join(label_root_path, label_list[i]))
    #     h, w, c = label.shape
    #     label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    #     # print(np.max(label), np.min(label))
    #     thresh = 20
    #     _, label = cv2.threshold(label, thresh, 255, cv2.THRESH_BINARY)
    #     save_root = '../_raw_data/train/labels_binary/'
    #     if not os.path.exists(save_root):
    #         os.mkdir(save_root)
    #     save_fn = os.path.join(save_root, '%s.png' % (label_list[i].split('.')[0]))
    #     cv2.imwrite(save_fn, label)

    ######## Crop images ########
    # if i == 2:
    #     break

    image = cv2.imread(os.path.join(image_root_path, image_list[i]))
    label = cv2.imread(os.path.join(label_root_path, label_list[i]), flags=cv2.IMREAD_UNCHANGED)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

    if resize:
        image = cv2.resize(image, (resize_w, resize_h))
        label = cv2.resize(label, (resize_w, resize_h))

    h, w, c = image.shape
    H = int(np.ceil((h - sub_size) / stride) + 1)
    W = int(np.ceil((w - sub_size) / stride) + 1)
    # convert to binary mask
    thresh = 20
    _, label = cv2.threshold(label, thresh, 255, cv2.THRESH_BINARY)

    for j in range(H):  # row
        for k in range(W):  # column
            if j != H - 1 and k != W - 1:  # coordinates of lower-right corner
                coord = (j * stride + sub_size, k * stride + sub_size)
            elif j == H - 1 and k != W - 1:
                coord = (h, k * stride + sub_size)
            elif j != H - 1 and k == W - 1:
                coord = (j * stride + sub_size, w)
            else:
                coord = (h, w)
            sub_image = image[coord[0] - sub_size:coord[0], coord[1] - sub_size:coord[1]]
            sub_label = label[coord[0] - sub_size:coord[0], coord[1] - sub_size:coord[1]]

            ## DEBUG: if label has crack, then save the subimage, else randomly skip the subimage (80%)
            # if sub_label.sum() < 1 and random.random() > 0.2:
            #     pass

            save_root_1 = '../data/train/images_sub/'
            save_root_2 = '../data/train/labels_sub/'
            if not os.path.exists(save_root_1):
                os.makedirs(save_root_1)
            if not os.path.exists(save_root_2):
                os.makedirs(save_root_2)
            image_save_name = '%s_%04d_%04d.png' % (image_list[i].split('.')[0], coord[0], coord[1])
            image_save_name = os.path.join(save_root_1, image_save_name)
            label_save_name = '%s_%04d_%04d.png' % (label_list[i].split('.')[0], coord[0], coord[1])
            label_save_name = os.path.join(save_root_2, label_save_name)
            cv2.imwrite(image_save_name, sub_image)
            cv2.imwrite(label_save_name, sub_label)


if __name__ == '__main__':

    import multiprocessing
    import os

    image_root_path = '../_raw_data/train/images'
    label_root_path = '../_raw_data/train/labels'

    image_list = os.listdir(image_root_path)
    label_list = os.listdir(label_root_path)

    pool = multiprocessing.Pool(processes=6)
    pool.map(main, range(len(image_list)))
    pool.close()
    pool.join()




