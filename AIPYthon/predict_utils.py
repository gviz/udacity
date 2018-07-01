import argparse
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from torch import nn
from model_map import model_map
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

debug_model = True

def debug_func(func):
    def wrapper(*args):
        if debug_model:
            func(args)       
    return wrapper

@debug_func
def debug_log(*args):
    print(args)
    
def load_category_map(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
        debug_log(len(cat_to_name.keys()))
    return cat_to_name
    
def build_classifier(ip_layer, op_layer, num_hidden_units, hidden_unit):
    seq = []
    seq.append(nn.Linear(ip_layer[0], ip_layer[1]))
    seq.append(nn.ReLU())
    seq.append(nn.Dropout(p = 0.5))
    if num_hidden_units:
        for unit in range(num_hidden_units):
            seq.append(nn.Linear(hidden_unit[0],hidden_unit[1]))
            seq.append(nn.ReLU())
            seq.append(nn.Dropout(p = 0.5))
    seq.append(nn.Linear(hidden_unit[1], op_layer[0]))  
    seq.append(nn.ReLU())
    seq.append(nn.Dropout(p = 0.5))
    seq.append(nn.Linear(op_layer[0], op_layer[1]))              
    seq.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*seq)                           
            
def load_checkpoint(filepath):
    """
        Load saved model , rebuild the classifier and attach it to the model.
    """
    checkpoint = torch.load(filepath)
    mdl = checkpoint['arch']
    model = model_map[mdl]['model'](pretrained = True)
    model.class_to_idx = checkpoint['class_to_idx']
    ip_layer = checkpoint['ip_layer']
    op_layer = checkpoint['op_layer']
    num_hidden_units = checkpoint['num_hidden_layers']
    hidden_layer = checkpoint['hidden_layer']
        
    model.classifier = build_classifier(ip_layer, op_layer, num_hidden_units, hidden_layer)
    #if debug_model:
    #    print(model)
    debug_log(model)
    model.load_state_dict(checkpoint['state_dict'])
    #if debug_model:
    #    print(model)
    debug_log(model)
    return model

def get_idx_to_class(model):
    return {val: key for key, val in model.class_to_idx.items()}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    img = Image.open(image)
    
    scale_factor = 1
    
    width = img.width
    height = img.height
    
    if width < height:
        if width >= 256:
            scale_factor = 256/width
    else:
        if height >= 256:
            scale_factor = 256/height
      
    debug_log("Before Resize: width: {} Height:{}\n".format(width, height))
    img = img.resize((int(width * scale_factor), int(height * scale_factor)))
      
    width = img.width
    height = img.height
    debug_log("Width: {} Height: {}\n".format(width,height))
    #print("Width: {} Height: {}\n".format(width,height))
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
 
    np_image = np.array(img, dtype = np.float)
    np_image = np_image/255.0
 
# Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = (np_image - mean)/std
    image = image.transpose((2, 0, 1))
    img = torch.from_numpy(image)
    return img



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if(gpu):
        model.cuda()
    image = process_image(image_path).float()
    with torch.no_grad():
        image = Variable(image.unsqueeze(0), volatile = True)
        debug_log("Imagesize {}".format(image.size()))
        if(gpu):
            image = image.cuda()
        model.eval()
        output = model.forward(image)
        ps = torch.exp(output).data
    return(image,ps.cpu().topk(topk))
