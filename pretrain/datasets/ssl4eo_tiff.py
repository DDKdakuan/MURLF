import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
import time


class TiffDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_paths = []
        # self.image_paths = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder) if filename.endswith('.tiff')]

        output_file = r"/share/home/aitlong/DDK/Datasets/image_paths_ssl4eo.txt"
        # with open(output_file, "w") as f:
        #     for item in self.image_paths:
        #          f.write("%s\n" % item)
        # print("列表已保存为:", output_file)
        with open(output_file, "r") as f:
            for line in f:
                item = line.strip() 
                self.image_paths.append(item)

    def __len__(self):
        return len(self.image_paths)
   

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_data = self.read_image_gdal(image_path)

        # image_tensor = torch.from_numpy(image_data)

        if self.transform is not None:
            image_tensor = self.transform(image_data)

        return image_tensor

    def read_image_gdal(self, image_path):

        dataset = gdal.Open(image_path)
        if dataset is None:
            raise Exception("Failed to open the image.")

        image_data = []
        for band in range(dataset.RasterCount):
            band_data = dataset.GetRasterBand(band + 1).ReadAsArray()
            image_data.append(band_data)

        image_data = np.stack(image_data, axis=-1)

        image_data1 = np.concatenate((image_data[:,:,:2],image_data[:,:,3:-3],image_data[:,:,-2:]),axis=-1)
        return image_data1

#

if __name__ == '__main__':

    import torchvision.transforms as cvtransforms
    import numpy as np
    import torch
    import random
    import cv2

    train_transforms_s1 = cvtransforms.Compose([

        cvtransforms.ToTensor(),
        cvtransforms.RandomResizedCrop(224, scale=(0.8, 1.)),  # 支持tensor与pil
        cvtransforms.RandomHorizontalFlip()  # 支持tensor与pil
    ])

    train_dataset = TiffDataset(
        # lmdb_file='/p/scratch/hai_dm4eo/wang_yi/data/ssl4eo_50k.lmdb',
        data_folder=r'/share/home/aitlong/DDK/Datasets/ssl4eotiff',
        transform=train_transforms_s1
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=8)
    # print(len(train_dataset))






