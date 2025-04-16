import torch
import os
from cv2 import imread
import scipy.io as scio
import numpy as np
import scipy.io
import h5py
import mat73


class HSIdataset(torch.utils.data.Dataset):
    """ Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """
    def __init__(self, root, split='train'):
        super().__init__()

        if split in ['train_1', 'train_2', 'val', 'test2']:
            self.img_pth = os.path.join(root, split)
        else:
            raise Exception('Unknown split {}'.format(split))

        self.image_list = self._get_image_list(split)

    def _get_image_list(self, split):
        if split == 'train_1':
            image_list = ['{:d}.mat'.format(i) for i in range(1, 1472)] #1472
        elif split == 'train_2':
            image_list = ['{:d}.mat'.format(i) for i in range(1, 720)]  #720
        elif split == 'val':
            image_list = ['{:d}.mat'.format(i) for i in range(1, 3)] #1320
        elif split == 'test2':
            image_list = ['{:d}.mat'.format(i) for i in range(8, 9)] #3236
        else:
            raise Exception

        return image_list

    def is_hdf5_file(self,file_path):
        # 使用 h5py 检查文件是否为 HDF5 格式
        try:
            with h5py.File(file_path, 'r') as f:
                # 如果文件能够以 HDF5 格式打开，就返回 True
                return True
        except OSError:
            # 如果不是 HDF5 文件格式，返回 False
            return False
    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        #print(self.image_list[im_id])
        '''try:
            if self.is_hdf5_file(path):
                mat = mat73.loadmat(path)
            else:
                mat = scipy.io.loadmat(path)
            mat = mat['blockData']
            HSI = mat.astype(np.float32)
            return HSI
        except (OSError, KeyError) as e:
            # 如果读取文件时出错，捕获异常并输出错误信息
            print(f"Error reading file {self.image_list[im_id]}: {e}")
            return None  # 返回 None 跳过这个文件
        return HSI'''
        if self.is_hdf5_file(path):
            mat = mat73.loadmat(path)
        else:
            mat = scipy.io.loadmat(path)
        mat = mat['blockData']
        HSI = mat.astype(np.float32)
        return HSI

    def get_image(self, im_id):
        frame = self._get_image(im_id)

        return frame

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        frame = self._get_image(index)
        '''while frame is None:
            print(f"Skipping index {index} due to read error.")
            index = (index + 1) % len(self)  # 避免索引超出范围
            frame = self._get_image(index)'''

        return frame  # 返回有效的 frame

