import torch
import numpy as np
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Allowed colours
hairClassesList =  ['pink', 'blue','orange',  'white','aqua', 'gray','purple', 
                  'black', 'green', 'red','blonde','brown']
eyeClassesList= ['pink','red','black', 'orange' , 'purple', 'yellow', 'aqua', 'green', 
               'brown', 'blue']



def storeModelParams(modRef, optRef, step, log, location):

    state = {'model' : modRef.state_dict(),
             'optim' : optRef.state_dict(),
             'step' : step,
             'log' : log}

    torch.save(state, location)
    return

def denorm(image):	
    denormForm = image / 2 + 0.5
    return denormForm.clamp(0, 1)

def getModelParams(model, optimizer, file_path):

    loadPrev = torch.load(file_path)
    
    optimizer.load_state_dict(loadPrev['optim'])
    model.load_state_dict(loadPrev['model'])
    
    resumeFromStep = loadPrev['step']
    log = loadPrev['log']
    
    return model, optimizer, resumeFromStep, log


def plot_loss(g_log, d_log, file_path):

    steps = list(range(len(g_log)))
    plt.semilogy(steps, g_log)
    plt.semilogy(steps, d_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return


def getRandomLable(batch, hair_cl, eye_cl):
	
    hair_type = np.random.choice(hair_cl, batch) 
    eye_type = np.random.choice(eye_cl, batch)
    
    hCode = torch.zeros(batch, hair_cl)  
    eCode = torch.zeros(batch, eye_cl)    
    
    for i in range(batch):
        hCode[i][hair_type[i]] = 1
        eCode[i][eye_type[i]] = 1

    return torch.cat((hCode, eCode), dim = 1) 

def generateByHairEye(modelRef, device, lD, hairClasses, eyeClasses, 
    Dpath, step = None, hairColor = None, eyeColor = None):
    weight=1
    bias=0
    vecSize=64

    hairColorVec = torch.zeros(vecSize, hairClasses).to(device)
    eyeColorVec = torch.zeros(vecSize, eyeClasses).to(device)
    hairClass = np.random.randint(hairClasses)
    eyeClass = np.random.randint(eyeClasses)

    for i in range(vecSize):
    	hairColorVec[i][hairClass]=1
	eyeColorVec[i][eyeClass] = 1

    concate = torch.cat((hairColorVec, eyeColorVec), 1)
    z = torch.randn(vecSize, lD).to(device)

    output = modelRef(z, concate)
    save_image(denorm(output), os.path.join(Dpath, '/{} hair {} eyes.png'.format(hairColor,eyeColor)))
