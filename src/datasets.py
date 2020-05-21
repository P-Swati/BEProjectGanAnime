import torch
import pickle
import os
import cv2
import numpy as np

        
class DataLoad:

    def __init__(self, root, tagsPickle, transFunc):
        try:
        with open(tagsPickle, 'rb') as file:
            self.tagsPickle = pickle.load(file) 
        except:
            print("pickle file not found")
        self.root = root
        self.images = os.listdir(self.root)
        self.transFunc = transFunc
        self.fileType='.jpg'
        self.lengthOfDataset = len(self.images)
        
    
    
    def getTuple(self, row): 
        ipath = os.path.join(self.root, str(row) + self.fileType)
        hairClass, eyeClass = self.tags_file[row]
        pic = cv2.imread(ipath)
        # (BGR -> RGB)
        pic = pic[:, :, (2, 1, 0)]    						 
        if self.transFunc:
            finalPic = self.transFunc(pic)
        return finalPic, hairClass, eyeClass

    def length(self):
        return self.lengthOfDataset

class RandomBatchGetter:
    def __init__(self, data, batch):
        self.data = data
        self.lenOfDataset = self.data.length()
        self.bSize = batch
    
    def getDataBatch(self):
        indexList = np.random.choice(self.lenOfDataset, self.bSize)
        imageBatch = list()
        hairClassVec=list()
        eyeClassVec =list()
        for idx in indexList:
            img, hairClass, eyeClass = self.data.getTuple(idx)
            imgUnsq= img.unsqueeze(0)
            hairUnsq= hairClass.unsqueeze(0)
            eyeUnsq= eyeClass.unsqueeze(0)
                
            imageBatch.append(imgUnsq)
            hairClassVec.append(hairUnsq)
            eyeClassVec.append(eyeUnsq)
        
        imageBatch = torch.cat(imageBatch, 0)
        hairClassVec = torch.cat(hairClassVec, 0)
        eyeClassVec = torch.cat(eyeClassVec, 0)
        
        return imageBatch, hairClassVec, eyeClassVec
    
