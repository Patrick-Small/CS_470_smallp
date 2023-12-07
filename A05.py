##############
#Assignment 05
#CS-470
#Patrick Small
##############

#############################################################################
#With assistance from:
#https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
#############################################################################


#Imports (most taken directly from TorchLand.py)
import torch
from torch import nn 
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import get_model, get_weight
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


#Class that handles the create_module part
class Network(nn.Module):
        
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32,32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )
        
    def forward(self, x):        
        logits = self.net_stack(x)
        return logits
        

#Returns a list of the names of all combinations you will be testing
def get_approach_names():
    
    #Return the list of names
    return ["VGG19", "approach_2"]


#Given the name, return a description of what makes this approach distinct.
def get_approach_description(approach_name):
    
    if approach_name == 'VGG19':
        
        return "I don't know yet"
    
    elif approach_name == 'approach_2':
        
        return "I also don't know yet"
    
    else:
        
        return "Unknown entry"


#Given the name and whether it's for training data, return the appropriate dataset transform.
#Does this need to use approach_name???
def get_data_transform(approach_name, training):
    
    #return v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]);
    
    match approach_name:
        
        case "VGG19":
            
            weights = get_weight ("VGG19_Weights.DEFAULT")
            preprocess = weights.transforms()
            data_transform = preprocess
            return data_transform
                
        case "approach_2":
            
            #If it's for training
            if training:
        
                data_transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]) 
        
            #If it's not for training
            else:
        
                data_transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
                    
            return data_transform
        

#Given the name, return the prefered batch size. Whatever you want it to be!
def get_batch_size(approach_name):
    
    if approach_name == 'VGG19':
        
        return 32
    
    elif approach_name == 'approach_2':
        
        return 32


#Given the name and output, build and return a Pytorch neural network
def create_model(approach_name, class_cnt):
    
    if approach_name == "VGG19":
        model = get_model("vgg19", weights = "DEFAULT")
    
        for param in model.parameters():
            param.requires_grad = False
        
        print("BEFORE:")
        print(model)
    
        feature_cnt = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(feature_cnt, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    
        print("AFTER:")
        print(model)
        
        return model
    
    elif approach_name == "approach_2":
        
        return Network(class_cnt)


#Actually training the model
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    
    print("HELLO")
    
    if approach_name == "VGG19":
        print(approach_name)
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Move the model to the specified device
        model.to(device)

        epochs = 1 #10
        print("About to start epoch loop")
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            print("** EPOCH", (epoch+1), "**")

            for index, (inputs, labels) in enumerate(train_dataloader):
                if index % 10 == 0:
                    print("Batch", index, "...")
                    
                #print("About to enter batch...")
                inputs, labels = inputs.to(device), labels.to(device)
                  
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                              
                # Zero the gradients
                optimizer.zero_grad()


                running_loss += loss.item()

            # Print the average training loss for this epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader)}")

        # Validation loop
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate and print accuracy on the test set
        #accuracy = accuracy_score(all_labels, all_preds)
        #print(f"Accuracy on test set: {accuracy * 100:.2f}%")
        

    
    #Return the trained version of the model
    return model
