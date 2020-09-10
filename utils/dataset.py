import os
from os.path import splitext
from os import listdir
import numpy as np
from torch.utils.data import Dataset
import logging
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as tvF
import random
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# import kornia
import time


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        # # DEBUG
        # self.ids = self.ids[:1024]
        self.ids.sort()

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):
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

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    def img_distortion(self, im, im_mask):
        # img : (0,255)  imgmask(0,1)
        imshape = im.shape
        im_maskshape = im_mask.shape
        if len(imshape) == 2:
            im = im[..., np.newaxis]
            imshape = im.shape
        if len(im_maskshape) == 2:
            im_mask = im_mask[..., np.newaxis]
        im_merge = np.concatenate((im, im_mask), axis=2)
        im_merge_t = self.elastic_transform(im_merge, im_merge.shape[1] * 10, im_merge.shape[1] * 0.1,
                                       im_merge.shape[1] * 0.1)
        imgout = im_merge_t[..., :imshape[2]].astype('uint8')
        maskout = im_merge_t[..., imshape[2]:]

        return imgout, maskout

    def gaussian_blur(self, img, ks=27, sigma=10):
        image = np.array(img)
        image_blur = cv2.GaussianBlur(image, (ks, ks), sigma)
        image_blur = Image.fromarray(image_blur)
        return image_blur

    def transform(self, image, mask=None):
        w, h = image.size
        newW, newH = int(self.scale * w), int(self.scale * h)

        # Resize
        resize = transforms.Resize(size=(int(1.25*newW), int(1.25*newH)))
        image = resize(image)
        mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = tvF.hflip(image)
            mask = tvF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = tvF.vflip(image)
            mask = tvF.vflip(mask)

        # Random rotation
        angle = (random.random() - 0.5) * 60  # -30 ~ 30
        image = tvF.rotate(image, angle)
        mask = tvF.rotate(mask, angle)

        # Random Gaussian noise
        if random.random() > 0.75:
            ks_list = np.arange(7) * 2 + 5
            sigma_list = np.arange(5) + 3
            ks = np.random.choice(ks_list)
            sigma = np.random.choice(sigma_list)
            image = self.gaussian_blur(image, ks, sigma)

        # Random distortion
        mask_arr = np.array(mask)  # max value: 255
        if mask_arr.sum() > 1 and random.random() > 0.75:
            image_arr = np.array(image)
            image, mask = self.img_distortion(image_arr, mask_arr)
            image = tvF.to_pil_image(image)
            mask = tvF.to_pil_image(mask)
            # print(image.shape, mask.shape)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(newH, newH))
        image = tvF.crop(image, i, j, h, w)
        mask = tvF.crop(mask, i, j, h, w)

        # # DEBUG: check image
        # image.show()
        # print(image.size)

        # Transform to tensor
        image = tvF.to_tensor(image)
        mask = tvF.to_tensor(mask)

        # Normalize
        norm_img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = norm_img(image)

        return image, mask

    def __getitem__(self, i):
        # start = time.time()
        idx = self.ids[i]
        ## DEBUG: very very slow!!!
        mask_file = os.path.join(self.masks_dir, (idx + '.png'))
        img_file = os.path.join(self.imgs_dir, (idx + '.png'))
        mask = Image.open(mask_file)
        img = Image.open(img_file)
        # print(time.time() - start)

        img, mask = self.transform(img, mask)

        return {
            'image': img,
            'mask': mask
        }
