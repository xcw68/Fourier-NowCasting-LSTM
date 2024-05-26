import os
import cv2
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Radar(Dataset):
    def __init__(self, data_type, data_root='train'):
        self.data_type = data_type
        self.data_root = data_root  # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root, self.data_type)))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root, self.data_type, self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_' + str(i + 1) + '.png'
            img_path = os.path.join(cur_fold, file)
            img = cv2.imread(img_path, 0)[:, :, np.newaxis]
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32) / 255.0
        if self.data_type == 'test':
            return imgs, self.dirs[index]
        else:
            return imgs
