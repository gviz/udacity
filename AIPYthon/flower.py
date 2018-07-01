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
debug_model = False

class flower(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arch = args.arch
        self.model, self.ip_layer,self.hidden_unit, self.op_layer = self.GetModel(args.arch)
        self.gpu = args.GPU
        self.data_dir = args.data_dir
        self.train_dir = self.data_dir + '/train'        
        self.valid_dir = self.data_dir + '/valid'
        self.test_dir = self.data_dir + '/test'   
        self.save_dir = args.save_dir
        self.epoch = args.epochs
        self.num_hidden_units = args.hidden_units
        self.num_training_samples = 0
        self.training_accuracy = 0
        self.validation_accuracy = 0
        self.testing_accuracy = 0
        self.training_loss = 0
        self.validation_loss = 0
        self.testing_loss = 0
        self.lr = args.lr
        self.load_data()
        self.init_model()
        
    def save(self):
        """Save Models to file"""
        file_name = "flowers-{}.pth".format(self.arch)
        self.model.class_to_idx = self.train_dataset.class_to_idx
        checkpoint = {
                      'arch': self.arch,
                      'GPU': self.gpu,
                      'class_to_idx': self.model.class_to_idx,
                      'epochs': self.epoch,
                      'hidden_layer':  self.hidden_unit,
                      'num_hidden_layers': self.num_hidden_units,
                      'lr': self.lr,
                      'ip_layer': self.ip_layer,
                      'op_layer': self.op_layer,
                      'state_dict': self.model.state_dict(),
                      'opt_dict': self.optimizer.state_dict()
                    }
        torch.save(checkpoint, self.save_dir + "/" + file_name)
        print("Saved model to {} ...\n".format(self.save_dir + "/" + file_name))
    
    def info(self):
        print("Model: Flower\n Arch: {}\n Data Dir: {}\n Training Dir: {}\n Validation Dir: {}\n"
              " Testing Dir: {}\n Epochs: {} Hidden Layers: {}\n"
              " LR: {}\n"
              " Training Images: {} Validation Samples: {} Testing Samples: {}".format(
              self.arch, self.data_dir, self.train_dir, self.valid_dir, self.test_dir,
              self.epoch, self.num_hidden_units, self.lr, self.num_training_samples,
              self.num_validation_samples, self.num_testing_samples))
              
    def build_classifier(self):
        seq = []
        seq.append(nn.Linear(self.ip_layer[0], self.ip_layer[1]))
        #seq.append(self.ip_layer)
        seq.append(nn.ReLU())
        seq.append(nn.Dropout(p = 0.5))
        if self.num_hidden_units:
            for unit in range(self.num_hidden_units):
                seq.append(nn.Linear(self.hidden_unit[0],self.hidden_unit[1]))
                seq.append(nn.ReLU())
                seq.append(nn.Dropout(p = 0.5))
                
        seq.append(nn.Linear(self.hidden_unit[1], self.op_layer[0]))   
        seq.append(nn.ReLU())
        seq.append(nn.Dropout(p = 0.5))
        seq.append(nn.Linear(self.op_layer[0], self.op_layer[1]))              
        seq.append(nn.LogSoftmax(dim=1))
        
        return nn.Sequential(*seq)                           
        
    def init_model(self):
        self.criterion = nn.NLLLoss()
        
        #Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = self.build_classifier()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = self.lr)
        if debug_model == True:
            print(self.model)
        self.model.train()
                           
    def show_results(self):
        print("=============Results==============")
        print("Training Samples : {}\n"
            "Validation Accuracy: {}\nTesting Accuracy: {}\n"
            "Training Loss: {}\n".format( self.num_training_samples,
                                        self.validation_accuracy,
                                        self.testing_accuracy,
                                        self.training_loss))
        
    def load_data(self):
        #For Validation and Testing
        test_transforms = transforms.Compose(
                    [transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor() ,
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
         #For Training
        train_transforms = transforms.Compose(
                    [transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ])
            # TODO: Load the datasets with ImageFolder
        self.train_dataset = datasets.ImageFolder(self.train_dir,
                                              transform = train_transforms)

        #Use same transforms for validation and testing
        self.validation_dataset = datasets.ImageFolder(self.valid_dir,
                                                   transform = test_transforms)
        self.test_dataset = datasets.ImageFolder(self.test_dir,
                                                 transform = test_transforms)
# TODO: Using the image datasets and the trainforms, define the dataloaders
        self.train_dataloaders = torch.utils.data.DataLoader(self.train_dataset,
                                                             batch_size=64,
                                                             shuffle=True)
        self.validation_dataloaders = torch.utils.data.DataLoader(self.validation_dataset,
                                                                  batch_size=10)
        self.test_dataloaders = torch.utils.data.DataLoader(self.test_dataset,
                                                            batch_size=10)
        self.num_training_samples = len(self.train_dataset)
        self.num_validation_samples = len(self.validation_dataset)
        self.num_testing_samples = len(self.test_dataset)
        
    def test_model(self, type):
        flowers = self.model  
        if(self.UseGPU()): 
            flowers.cuda()
        if type == "Validation":
              dl = self.validation_dataloaders              
        else:
              dl = self.test_dataloaders
              
        flowers.eval()
        count = 0
        accuracy = 0
        loss = 0
        print_sample = 0

        for iputs, lbls in dl:
            if count == 1 and debug_model == True:
              break
            
            with torch.no_grad():
                inputs, labels = Variable(iputs, volatile = True), Variable(lbls, volatile = True)
                if(self.UseGPU()):  
                    inputs, labels = inputs.cuda(), labels.cuda()
                output = flowers.forward(inputs)
                ps = torch.exp(output).data
                #loss += self.criterion(output, labels).data[0]
                loss += self.criterion(output, labels).item()
                #print("loss: {}".format(loss.data))
                pred = ps.max(1)[1][0]
                if print_sample:
                    print("Prediction: {} {}: label: {}\n".format(pred,cat_to_name[str(pred + 1)], cat_to_name[str(labels.data[0]+1)]))

                equality = (ps.max(1)[1] == labels.data)

                accuracy += equality.type_as(torch.FloatTensor()).mean()
                #print("accuracy {}".format(accuracy))
                count += 1

        if type == "Validation":        
              total_accuracy = self.validation_accuracy = accuracy/len(dl)
              total_loss = self.validation_loss = loss/len(dl)
        else:             
              total_accuracy = self.testing_accuracy = accuracy/len(dl)
              total_loss = self.testing_loss =  loss/len(dl)
              
        print("{} Done : {} samples tested\n".format(type, count))
        print("{} Loss: {:.3f}".format(type, total_loss ))
        print("{} Accuracy: {:.3f}".format(type, total_accuracy))
        
    def validate(self):
        self.test_model("Validation")
    
    def test(self):
        self.test_model("Testing")
              
    def train(self):
        flowers = self.model    
        print(flowers)
        #return
        if(self.UseGPU()):
              flowers.cuda()
        flowers.train()
        count = 0
        train_loss = 0
        print_sample = 0

        for e in range(self.epoch):
            for indx, (inputs, labels) in enumerate(self.train_dataloaders):
                count += 1
               # if(count == 1 and debug_model == True):
               #     break
                input_t, label_t = Variable(inputs), Variable(labels)
                if(self.UseGPU()):
                    input_t, label_t = input_t.cuda(), label_t.cuda()
                self.optimizer.zero_grad()
                output = flowers.forward(input_t)
                loss = self.criterion(output, label_t)
                #train_loss += loss.data[0]
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            #do Validation for every Epoch
            flowers.eval()
            count = 0
            val_accuracy = 0
            val_loss = 0
            print_sample = 0

            for iputs, lbls in self.validation_dataloaders:
                with torch.no_grad():
                    inputs, labels = Variable(iputs, volatile = True), Variable(lbls, volatile = True)
                    inputs, labels = inputs.cuda(), labels.cuda()
                    logbits = flowers.forward(inputs)
                    ps = torch.exp(logbits).data
                    val_loss += self.criterion(logbits, labels).item()
                    #print("loss: {}".format(loss.data))
                    pred = ps.max(1)[1][0]
                    equality = (ps.max(1)[1] == labels.data)

                    val_accuracy += equality.type_as(torch.FloatTensor()).mean()
                    #print("accuracy {}".format(accuracy))
                    count += 1

            print("Validation Done : {}\n".format(count))
            print("Validation Loss: {:.3f}".format(val_loss/len(self.validation_dataloaders)))
            print("Validation Accuracy: {:.3f}".format(val_accuracy/len(self.validation_dataloaders)))


        #Save Model  
        print("Saving Trained Model ....")
        if self.save_dir != None:
              self.save()        
      
        self.training_loss =   train_loss/len(self.train_dataloaders)
        
        print("Training Done : {}....\n".format(len(self.train_dataloaders)))
        print("Training Loss: {}".format(self.training_loss))  
        
        
    def forward(self, data):
        self.model.forward(data)
        
    def UseGPU(self):
        return self.gpu
    
    def GetModel(self, arch):
        if arch not in model_map.keys():
            return None
        else:
            return (
                    model_map[arch]['model'](pretrained = True),
                    model_map[arch]['ip_layer'],
                    model_map[arch]['hidden_layer'],
                    model_map[arch]['op_layer']
                   )
    
