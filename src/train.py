import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os

from datasets import DataLoad, RandomBatchGetter
from ACGAN import MyConGANGen, MyConDisc
from utils import plot_loss, storeModelParams, denorm
from utils import generateByHairEye, getRandomLable

device='cuda'

def mul(a,b):
    return a*b

def main():
    batch_size = 128
    iterations =  40000
    device='cuda'
    
    hairClassCount= 12
    eyeClassCount=10
    totalNumOfClasses = hairClassCount + eyeClassCount
    latentVecDim = 100
    
    print("Batch Size : ",batch_size)
    print("Iterations : ",iterations)
   
    root='../content/images'
    tags='../content/features.pickle'
   
    resultsDir = '../content/results'
    modelDir = '../content/model'
        
    ########## Training Code ##########

    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = DataLoad(root = root, tagsPickle = tags, transFunc = transform)
    RandomBatchGetter = RandomBatchGetter(data = dataset, batch = batch_size)
    
    D = MyConDisc(countOfClasses = totalNumOfClasses).to(device)
    G = MyConGANGen(latentVectorSize = latentVecDim, classVectorSize = totalNumOfClasses).to(device)
    

    optimGen = optim.Adam(G.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    optimDisc = optim.Adam(D.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    
    d_losses=list()
    g_losses=list()
    lossFunc = torch.nn.BCELoss()
   
    # start of traning loop
    print("training loop start..")
    
    for curIteration in range(1, iterations + 1):

        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        
        ############# Train discriminator ########################################
        #fake batch
        z = torch.randn(batch_size, latentVecDim).to(device)
        
        fakeTag = getRandomLable(batch_size = batch_size, 
                                    hair_classes = hair_classes,
                                    eye_classes = eye_classes).to(device)
        
        fakeImage = G(z, fakeTag).to(device)
        
        print("real batch")
        realImage, hairClasses, eyeClasses = RandomBatchGetter.getDataBatch()
        
        realImage= realImage.to(device) 
        hairClasses= hairClasses.to(device)
        eyeClasses=eyeClasses.to(device)
        
        realTag = torch.cat((hairClasses, eyeClasses), dim = 1)
        
        print("actual tag of image"+str(realTag))
                
         #pass through D
        realProbab, realMultiPredict = D(realImage)
        fakeProbab, fakeMultiPredict = D(fakeImage)
            
        real_discrim_loss = lossFunc(realProbab, 1)
        fake_discrim_loss = lossFunc(fakeProbab, 0)

        realClassifierLoss = lossFunc(realMultiPredict, realTag)
        
        totalDiscrimloss = (real_discrim_loss + fake_discrim_loss) * 0.5
        totalclassificationLayerLoss = mul(realClassifierLoss ,classification_weight)
        
        classifier_log.append(classifier_loss.item())
       
        # Train generator
        z = torch.randn(batch_size, latentVecDim).to(device)
        fakeTag = getRandomLable(batch_size = batch_size, hair_classes = hair_classes, eye_classes = eye_classes).to(device)
        
        fakeImage = G(z, fakeTag).to(device)
        
        DLoss = totalDiscrimloss + totalclassificationLayerLoss
        optimDisc.zero_grad()
        DLoss.backward()
        optimDisc.step()
        
        fake_score, fake_predict = D(fakeImage)
        
        discrim_loss = lossFunc(fake_score, real_label)
        classifier_loss = lossFunc(fake_predict, fakeTag)
        
        GLoss = classifier_loss + discrim_loss
        optimGen.zero_grad()
        GLoss.backward()
        optimGen.step()

        ########## save model configuration ##########

        if curIteration == 1:
            save_image(denorm(realImage[:64,:,:,:]), os.path.join('../content/saveimage/', 'real.png'))
        if curIteration % 500 == 0:
            save_image(denorm(fakeImage[:64,:,:,:]), os.path.join('../content/saveimage/', 'fake{}.png'.format(curIteration)))
            
        if curIteration % 2000 == 0:
            storeModelParams(modRef = G, optRef = optimGen, step = curIteration, log = tuple(g_losses), 
                       location = os.path.join('../content/checkpoint', 'G_{}.ckpt'.format(curIteration)))
            storeModelParams(modRef = D, optRef = optimDisc, step = curIteration, log = tuple(d_losses), 
                       location = os.path.join('../content/checkpoint', 'D_{}.ckpt'.format(curIteration)))
            
            plot_loss(g_log = g_losses, d_log = d_losses, file_path = os.path.join('../content/lossplot/', 'loss.png'))
            
            generateByHairEye(model = G, device = 'cuda', step = curIteration, latent_dim = latentVecDim, 
                                     hair_classes = hair_classes, eye_classes = eye_classes, 
                                     sample_dir = '../content/sample/')
    
if __name__ == '__main__':
    main()
