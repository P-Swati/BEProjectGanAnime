import torch
import torch.nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.transforms as Transform
import folium
from folium import plugins
from PIL import Image
import os
import numpy as np
import utils
import datasets
import ACGAN
import anvil.media

hair_mapping =  ['orange', 'white','aqua', 'gray', 'green', 'red', 'purple', 
                 'pink', 'blue', 'black', 'brown','blonde']
eye_mapping = ['black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 
               'brown', 'red', 'blue']

hair_dict = {
    'orange' : 0,
    'white': 1, 
    'aqua': 2,
    'gray': 3,
    'green': 4,
    'red': 5,
    'purple': 6,
    'pink': 7,
    'blue': 8,
    'black': 9,
    'brown': 10,
    'blonde': 11
}

eye_dict = {
    'black': 0,
    'orange': 1,
    'pink': 2,
    'yellow': 3,
    'aqua' : 4,
    'purple': 5,
    'green': 6,
    'brown': 7,
    'red': 8,
    'blue': 9
}

def generateUsingHairEye(model, device, hair_classes, eye_classes, lDim, hColor, eColor):
    htag = torch.zeros(64, hair_classes).to(device)
    etag = torch.zeros(64, eye_classes).to(device)
    hairLabelIndex = hair_dict[hColor]
    eyeLabelIndex = eye_dict[eColor]
    for i in range(64):
        htag[i][hairLabelIndex]=1
        etag[i][eyeLabelIndex] = 1
    
    fulltag = torch.cat((htag, etag), 1)
    z = torch.randn(64, lDim).to(device)
    
    output = model(z, tag)
    save_image(utils.denorm(output), '../generated/{} hair {} eyes.png'.format(hair_mapping[hairLabelIndex], eye_mapping[eyeLabelIndex]))

def add(hair,eye):
    return hair+eye

@anvil.server.callable   
def main(hairColor,eyeColor):
    if not os.path.exists('../generated'):
        os.mkdir('../generated')
    hairClasses = 12
    eyeClasses = 10
    batch_size = 1
    totalClasses = add(hairClasses,eyeClasses)
    latentDim = 100
    
    G_path = '/content/BeProjectGANAnime/src/mymodel/ACGAN_generator.ckpt'

    G = ACGAN.MyConGANGen(latentVectorSize = latent_dim, classVectorSize = totalClasses)
    previouslyTrainedstate = torch.load(G_path)
    
    print("load state info..")
    G.load_state_dict(previouslyTrainedstate['model'])
    G = G.eval()

    generateUsingHairEye(G, 'cpu',hairClasses, eyeClasses, latentDim,hairColor, eyeColor)

    return(anvil.media.from_file('/content/BeProjectGANAnime/generated/{} hair {} eyes.png'.format(hair,eye)))
