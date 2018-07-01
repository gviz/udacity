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

parser = argparse.ArgumentParser(description = "AI Trainer")
parser.add_argument('data_dir')
parser.add_argument('--save_dir', action = "store", dest = "save_dir")
parser.add_argument('--arch', action = "store", dest = "arch", default = "vgg16")
parser.add_argument('--learning_rate',action = "store", dest = "lr", type = float, default = .0001)
parser.add_argument('--epochs',action = "store", dest = "epochs", type = int, default = 4)
parser.add_argument('--hidden_units', action = "store", dest = "hidden_units", type = int, default = 1)
parser.add_argument('--GPU', action = "store_true", default = False)
args = parser.parse_args()
print(args)

model = flower(args)
model.info()
model.train()
model.validate()
model.test()
model.show_results()



