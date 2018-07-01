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
from flower import flower
from predict_utils import *

parser = argparse.ArgumentParser(description = "AI Flower Identification")
parser.add_argument('input_file')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', action = "store", dest = "topk", type=int, default=5)
parser.add_argument('--category_names', action = "store", dest = "category_name", default = "cat_to_name.json")
parser.add_argument('--GPU', action = "store_true", default = False)
args = parser.parse_args()
print(args)
   
model = load_checkpoint(args.checkpoint)
img, prediction = predict(args.input_file, model, args.topk, args.GPU)

fid = prediction[0].numpy()[0]
probabilities = fid

if(args.category_name != None):
    category = load_category_map(args.category_name)
    idx_to_class = get_idx_to_class(model)
else:
    category = model.class_to_idx
    idx_to_class = get_idx_to_class(model)
    
    
fnames = [idx_to_class[x] for x in prediction[1].numpy()[0]]
names = [category[str(nm)] for nm in fnames]    
print(probabilities)
print(names)
print(zip(fnames, names))
print("Flower identified as {}\n".format(names[0]))
print("Top {} probabilities: \n".format(args.topk))
for (name, probability) in (zip(names, probabilities)):
    print("{} - {} \n".format(name, probability)) 


    




