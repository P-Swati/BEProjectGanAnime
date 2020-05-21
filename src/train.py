import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os

from datasets import DataLoad, RandomBatchGetter
from ACGAN import MyConGANGen, MyConDisc
from utils import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils import generation_by_attributes, get_random_label

device='cuda'

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
    

    G_optim = optim.Adam(G.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    D_optim = optim.Adam(D.parameters(), betas = [0.5, 0.999], lr = 0.0002)
    
    d_losses=list()
    g_losses=list()
    lossFunc = torch.nn.BCELoss()
   
    # start of traning loop
    print("training loop start..")
    
    for curIteration in range(1, iterations + 1):

        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        
        # Train discriminator
        real_img, hair_tags, eye_tags = RandomBatchGetter.getDataBatch()
        
        real_img= real_img.to(device) 
        hair_tags= hair_tags.to(device)
        eye_tags=eye_tags.to(device)
        
        real_tag = torch.cat((hair_tags, eye_tags), dim = 1)
        
        print("actual tag of image"+str(real_tag))
         
        #fake batch
        z = torch.randn(batch_size, latentVecDim).to(device)
        
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = hair_classes,
                                    eye_classes = eye_classes).to(device)
        
        fake_img = G(z, fake_tag).to(device)
                
         #pass through D
        realProbab, realMultiPredict = D(real_img)
        fakeProbab, fakeMultiPredict = D(fake_img)
            
        real_discrim_loss = lossFunc(realProbab, real_label)
        fake_discrim_loss = lossFunc(fakeProbab, fake_label)

        real_classifier_loss = lossFunc(realMultiPredict, real_tag)
        
        discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5
        classifier_loss = real_classifier_loss * args.classification_weight
        
        classifier_log.append(classifier_loss.item())
            
        D_loss = discrim_loss + classifier_loss
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # Train generator
        z = torch.randn(batch_size, latentVecDim).to(device)
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = hair_classes, hair_prior = hair_prior,
                                    eye_classes = eye_classes, eye_prior = eye_prior).to(device)
        fake_img = G(z, fake_tag).to(device)
        
        fake_score, fake_predict = D(fake_img)
        
        discrim_loss = lossFunc(fake_score, real_label)
        classifier_loss = lossFunc(fake_predict, fake_tag)
        
        G_loss = classifier_loss + discrim_loss
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
            
        ########## Updating logs ##########
        d_losses.append(D_loss.item())
        g_losses.append(G_loss.item())
        show_process(total_steps = iterations, step_i = curIteration,
        			 g_log = g_losses, d_log = d_losses, classifier_log = classifier_log)

        ########## Checkpointing ##########

        if curIteration == 1:
            save_image(denorm(real_img[:64,:,:,:]), os.path.join(random_sample_dir, 'real.png'))
        if curIteration % 500 == 0:
            save_image(denorm(fake_img[:64,:,:,:]), os.path.join(random_sample_dir, 'fake_step_{}.png'.format(curIteration)))
            
        if curIteration % 2000 == 0:
            save_model(model = G, optimizer = G_optim, step = curIteration, log = tuple(g_losses), 
                       file_path = os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(curIteration)))
            save_model(model = D, optimizer = D_optim, step = curIteration, log = tuple(d_losses), 
                       file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(curIteration)))
            
            plot_loss(g_log = g_losses, d_log = d_losses, file_path = os.path.join(checkpoint_dir, 'loss.png'))
            
            generation_by_attributes(model = G, device = args.device, step = curIteration, latent_dim = latentVecDim, 
                                     hair_classes = hair_classes, eye_classes = eye_classes, 
                                     sample_dir = fixed_attribute_dir)
    
if __name__ == '__main__':
    main()
