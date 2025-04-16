import os
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import random


class CameraArrayImage:
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}.png'.format(path), cv2.IMREAD_UNCHANGED)
        if im_raw.dtype == np.uint16:
            im_raw = im_raw.astype(np.int32)
        #im_raw = np.load('{}.npy'.format(path))
        im_raw = torch.from_numpy(im_raw)
        return CameraArrayImage(im_raw)

    def __init__(self, im_raw):
        self.im_raw = im_raw

    def get_image_data(self):
        im_raw = self.im_raw.float()
        return im_raw

    def shape(self):
        shape = (self.im_raw.shape[0], self.im_raw.shape[1])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[r1:r2, c1:c2]
        return CameraArrayImage(im_raw)

    def postprocess(self, return_np=True):
        # Convert to rgb
        im = torch.from_numpy(self.im_raw.astype(np.float32))
        im_out = im.clamp(0.0, 255)
        if return_np:
            im_out = im_out.numpy()
        return im_out

class BurstSRDataset(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """
    def __init__(self, root, burst_size=9, crop_sz=80, center_crop=False, random_flip=False, split='train'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val' or 'test'
        """
        assert burst_size <= 9, 'burst_sz must be less than or equal to 14'
        #assert split in ['train', 'val', 'test']

        root = root + '/'
        super().__init__()

        self.burst_size = burst_size
        self.crop_sz = crop_sz
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.root = root

        self.burst_list = self._get_burst_list()

    def _get_burst_list(self):
        burst_list = sorted(os.listdir('{}'.format(self.root)))
        #print(burst_list)
        return burst_list

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 9, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_raw_image(self, burst_id, im_id):
        raw_image = CameraArrayImage.load('{}/{}/LR_{:02d}'.format(self.root, self.burst_list[burst_id], im_id))
        return raw_image

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, info

    def _sample_images(self):

        '''ids = [1, 2, 3, 4, 6, 7, 8, 9]
        ids = [5, ] + ids # 5 is ref'''
        ids=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        return ids

    def __len__(self):
        return len(self.burst_list)

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 9 is used.
        im_ids = self._sample_images()

        # Read the burst images along with HR ground truth
        frames, meta_info = self.get_burst(index, im_ids)

        # Extract crop if needed
        if frames[0].shape()[-1] != self.crop_sz:
            if getattr(self, 'center_crop', False):
                r1 = (frames[0].shape()[-2] - self.crop_sz) // 2
                c1 = (frames[0].shape()[-1] - self.crop_sz) // 2
            else:
                r1 = random.randint(0, frames[0].shape()[-2] - self.crop_sz)
                c1 = random.randint(0, frames[0].shape()[-1] - self.crop_sz)
            r2 = r1 + self.crop_sz
            c2 = c1 + self.crop_sz
            frames = [im.get_crop(r1, r2, c1, c2) for im in frames]

        # Load the RAW image data
        burst_image_data = [im.get_image_data() for im in frames]
        if self.random_flip:
            if random.random() > 0.5:
                burst_image_data = [im.flip([1, ]).contiguous() for im in burst_image_data] #

            if random.random() > 0.5:
                burst_image_data = [im.flip([0, ]).contiguous() for im in burst_image_data]

        burst = torch.stack(burst_image_data, dim=0) #沿着一个新维度对输入张量序列进行连接
        burst = burst.float()
        return burst, meta_info['burst_name']


def pack_raw_image(im_raw):
    if isinstance(im_raw, np.ndarray):
        im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
    else:
        raise Exception

    im_out[0, :, :] = im_raw[0::2, 0::2]
    im_out[1, :, :] = im_raw[0::2, 1::2]
    im_out[2, :, :] = im_raw[1::2, 0::2]
    im_out[3, :, :] = im_raw[1::2, 1::2]
    return im_out


def flatten_raw_image(im_raw_4ch):
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2))
    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros((im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype)
    else:
        raise Exception

    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]

    return im_out
