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

hairClassesList =  ['pink', 'blue','orange',  'white','aqua', 'gray','purple', 
                  'black', 'green', 'red','blonde','brown']
eyeClassesList= ['pink','red','black', 'orange' , 'purple', 'yellow', 'aqua', 'green', 
               'brown', 'blue']

hair_dict={}
for i in range(len(hairClassesList)):
    hair_dict[hairClassesList[i]]=i

eye_dict = {}
for i in range(len(eyeClassesList)):
    eye_dict[eyeClassesList[i]]=i


def generateUsingHairEye(model, device, hairClasses, eyeClasses, lDim, hColor, eColor):
    vecSize=64
    htag = torch.zeros(vecSize, hairClasses).to(device)
    etag = torch.zeros(vecSize, eyeClasses).to(device)
    hairLabelIndex = hair_dict[hColor]
    eyeLabelIndex = eye_dict[eColor]
    for i in range(vecSize):
        htag[i][hairLabelIndex] = 1
        etag[i][eyeLabelIndex] = 1
    
    fulltag = torch.cat((htag, etag), 1)
    z = torch.randn(vecSize, lDim).to(device)
    
    output = model(z, fulltag)
    save_image(utils.denorm(output), '../generated/{} hair {} eyes.png'.format(hairClassesList[hairLabelIndex], eyeClassesList[eyeLabelIndex]))

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
    
    pathToModel = '/content/BeProjectGANAnime/src/mymodel/ACGAN_generator.ckpt'

    Gen = ACGAN.MyConGANGen(latentVectorSize = latent_dim, classVectorSize = totalClasses)
    previouslyTrainedstate = torch.load(pathToModel)
    
    print("load state info..")
    Gen.load_state_dict(previouslyTrainedstate['model'])
    Gen = Gen.eval()

    generateUsingHairEye(Gen, 'cpu',hairClasses, eyeClasses, latentDim,hairColor, eyeColor)

    return(anvil.media.from_file('/content/BeProjectGANAnime/generated/{} hair {} eyes.png'.format(hair,eye)))
