import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

# Allowed colours
hair_mapping =  ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 
                 'pink', 'blue', 'black', 'brown', 'blonde']

eye_mapping = ['black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 
               'brown', 'red', 'blue']

def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1])
    """
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)

def save_model(model, optimizer, step, log, file_path):

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step,
             'log' : log}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):

    prev_state = torch.load(file_path)
    
    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_step = prev_state['step']
    log = prev_state['log']
    
    return model, optimizer, start_step, log
    
def show_process(total_steps, step_i, g_log, d_log, classifier_log):

    print('Step {}/{}: G_loss [{:8f}], D_loss [{:8f}], Classifier loss [{:8f}]'.format(
            step_i, total_steps, g_log[-1], d_log[-1], classifier_log[-1]))
    return

def plot_loss(g_log, d_log, file_path):

    steps = list(range(len(g_log)))
    plt.semilogy(steps, g_log)
    plt.semilogy(steps, d_log)
    plt.legend(['Generator Loss', 'Discriminator Loss'])
    plt.title("Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return

def plot_classifier_loss(log, file_path):
    
    steps = list(range(len(log)))
    plt.semilogy(steps, log)
    plt.legend(['Classifier Loss'])
    plt.title("Classifier Loss ({} steps)".format(len(steps)))
    plt.savefig(file_path)
    plt.close()
    return

def get_random_label(batch_size, hair_classes, hair_prior, eye_classes, eye_prior):
    
    hair_code = torch.zeros(batch_size, hair_classes)  # One hot encoding for hair class
    eye_code = torch.zeros(batch_size, eye_classes)  # One hot encoding for eye class

    hair_type = np.random.choice(hair_classes, batch_size, p = hair_prior)  # Sample hair class from hair class prior
    eye_type = np.random.choice(eye_classes, batch_size, p = eye_prior)  # Sample eye class from eye class prior
    
    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eye_code[i][eye_type[i]] = 1

    return torch.cat((hair_code, eye_code), dim = 1) 

def generation_by_attributes(model, device, latent_dim, hair_classes, eye_classes, 
    sample_dir, step = None, fix_hair = None, fix_eye = None):
    
    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)
    hair_class = np.random.randint(hair_classes)
    eye_class = np.random.randint(eye_classes)

    for i in range(64):
    	hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1

    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)

    output = model(z, tag)
    if step is not None:
        file_path = '{} hair {} eyes, step {}.png'.format(hair_mapping[hair_class], eye_mapping[eye_class], step)
    else:
        file_path = '{} hair {} eyes.png'.format(hair_mapping[hair_class], eye_mapping[eye_class])
    save_image(denorm(output), os.path.join(sample_dir, file_path))
