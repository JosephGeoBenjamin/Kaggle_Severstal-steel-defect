'''
Utilities for working with Severstal Steel defect dataset

Dataset: https://www.kaggle.com/c/severstal-steel-defect-detection/

'''
import glob
import sys
import os
from torch.utils.data import Dataset
import numpy as np
import imageio as imio
import csv


class SeverstalSteelData(Dataset):
    ''' Used to load images and numpy GroundTruth
    '''
    def __init__(self, csv_file, root_dir = "../datasets/",
        imgExt=".jpg", gtExt=".npy", device='cpu'):
        self.device = device
        join = os.path.join
        self.imgList = data_path_fromCSV(join(root_dir,csv_file),
                                        join(root_dir,"images") ,
                                        dataExt = ".jpg")
        self.gtList = data_path_fromCSV(join(root_dir,csv_file),
                                        join(root_dir,"groundtruths") ,
                                        dataExt = ".npy")

        if (not self.imgList) or (not self.gtList):
            print("Empty data. Corruption on CSV read", file=sys.stderr)
        if (len(self.imgList) == len(self.gtList)):
            for i in range(len(self.imgList)):
                imgBase = os.path.basename(self.imgList[i]).replace(imgExt)
                gtBase = os.path.basename(self.gtList[i]).replace(gtExt)
                if ( imgBase != gtBase):
                    print("Corrupted: MisMatch in file names",imgList[i], gtList[i] ,file=sys.stderr)
        else:
            print("Corrupted: MisMatch in Image and GroundTruth count", file=sys.stderr)


    def __getitem__(self, idx):
        """ Returns torch format CHW
        """
        img = imio.read(imgList[idx])
        img = img.transpose(2,0,1)
        gt = np.load(gtList[idx])
        return img.to(self.device), gt.to(self.device)

    def data_path_fromCSV(csvFilePath, rootPath, dataExt = ".jpg"):
        imgNames = []
        with open(csvFilePath, "r") as csvFile:
            csv_reader = csv.DictReader(csvFile, delimiter=',')
            for lines in csv_reader:
                imgNames.append(lines['Image_Name'])

        dataPaths = []
        for name in imgNames:
            path = os.path.join(rootPath, name.replace(".jpg",dataExt))
            if os.path.isfile(path):
                dataPaths.append(path)
            else:
                print ("File Doesn't exist:" path)

        return dataPaths
