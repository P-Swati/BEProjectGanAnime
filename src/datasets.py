import torch
import pickle
import os
import cv2
import numpy as np

        
class DataLoad:

    def __init__(self, root, tagsPickle, transFunc):
        with open(tagsPickle, 'rb') as file:
            self.tagsPickle = pickle.load(file) 
        self.root = root
        self.transFunc = transFunc
        self.images = os.listdir(self.root)
        self.lengthOfDataset = len(self.images)
        self.fileType='.jpg'
    
    def length(self):
        return self.lengthOfDataset
    
    def getTuple(self, row):

        hairClass, eyeClass = self.tags_file[row]
        
        ipath = os.path.join(self.root, str(row) + self.fileType)
        pic = cv2.imread(ipath)
        # (BGR -> RGB)
        pic = pic[:, :, (2, 1, 0)]  
            						 
        if self.transFunc:
            finalPic = self.transform(pic)
        return finalPic, hairClass, eyeClass

class RandomBatchGetter:
    def __init__(self, data, batch):
        self.data = data
        self.bSize = batch
        self.lenOfDataset = self.data.length()
    
    def getDataBatch(self):
        indexList = np.random.choice(self.lenOfDataset, self.bSize)
        imageBatch, hairClassVec, eyeClassVec = [], [], []
        for idx in indexList:
            img, hairClass, eyeClass = self.data.getTuple(idx)
            imageBatch.append(img.unsqueeze(0))
            hairClassVec.append(hairClass.unsqueeze(0))
            eyeClassVec.append(eyeClass.unsqueeze(0))
        imageBatch = torch.cat(imageBatch, 0)
        hairClassVec = torch.cat(hairClassVec, 0)
        eyeClassVec = torch.cat(eyeClassVec, 0)
        
        return imageBatch, hairClassVec, eyeClassVec
    
