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

    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_step = prev_state['step']
    log = prev_state['log']
    
    return model, optimizer, start_step, log


def plot_loss(g_log, d_log, file_path):

    steps = list(range(len(g_log)))
    plt.semilogy(steps, g_log)
    plt.semilogy(steps, d_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return


def get_random_label(batch_size, hair_classes, eye_classes):
    
    hair_code = torch.zeros(batch_size, hair_classes)  
    eye_code = torch.zeros(batch_size, eye_classes)  

    hair_type = np.random.choice(hair_classes, batch_size)  # Sample hair class from hair class prior
    eye_type = np.random.choice(eye_classes, batch_size)  # Sample eye class from eye class prior
    
    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eye_code[i][eye_type[i]] = 1

    return torch.cat((hair_code, eye_code), dim = 1) 

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

    output = modelRef(z, tag)
    save_image(denorm(output), os.path.join(Dpath, '/{} hair {} eyes.png'.format(hairColor,eyeColor)))
