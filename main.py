# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:41:27 2020

@author: Anurag
"""


### Load required libraries
import os
import pandas as pd
import glob
import requests
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau


### Set working diectory
os.chdir('C:\\Users\\anurag\\Desktop\\dataset\\')

### Load train-dataset
df =  pd.DataFrame()
for i in range(2):
    names = [os.path.basename(f) for f in glob.glob('{}\\*.jpg'.format(i))]
    df1 =  pd.DataFrame({'display_url': names,'pav': i})
    df = df.append(df1, ignore_index=True)
del df1, names, i

### Load Json meta-data file
json_file = pd.read_json("pavbhaji.json")[['display_url','shortcode','is_video']]
json_file['display_url'] = json_file['display_url'].apply(lambda x: os.path.basename(x))

### Merge train and json dataframes
df1 = df.merge(json_file, how='outer', on='display_url')
del df, json_file

### Create test directory for storing scraped test files
if not os.path.exists('..\\test'):
    os.makedirs('..\\test')

### Remove video entries
df1 = df1[df1['is_video']==False].reset_index(drop=True)
df1 = df1.drop(columns=['is_video'])

### Download test set files from instragram 
for i in df1[df1['pav'].isnull()].index:
    u =  df1.loc[i,'shortcode']
    print('image number: {}'.format(i))
    result = requests.get('https://www.instagram.com/p/{}/'.format(u)).content
    soup = BeautifulSoup(result)
    metas = soup.find_all(attrs={"property": "og:image"})
    # print(metas[0].attrs['content'])
    print('len of metas: {}'.format(len(metas)))
    try:
        req =  requests.get(metas[0].attrs['content'])
        with open('..\\test\\' + df1['display_url'][i] , 'ab') as f:
            f.write(req.content)
    except:
        print(df1['display_url'][i])
        pass


### Data directory for train set
data_dir = r'C:\\Users\\anurag\\Desktop\\dataset'

### Load, trainsform data and split it into train and validation set
def load_split_train_test(data_dir,  b_size, valid_size =.1):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ColorJitter(0.1, contrast=0.1, saturation=0.1),
                                            # transforms.RandomRotation(90),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),])
    train_data = datasets.ImageFolder(data_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir,transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, 
                                              batch_size=b_size)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler,
                                             batch_size=b_size)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, b_size= 20, valid_size =.1)
print(trainloader.dataset.classes)

### Select GPU if available and load pretrained ResNet-50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
print(model)

### Turn off parameter learning for the pre-trained layers
for param in model.parameters():
    param.requires_grad = False
    # print(param.shape)

### Attach new custom fully connected layers at the end of connvoluional layers 
model.fc = nn.Sequential(nn.Linear(2048, 1000),
                         nn.ReLU(),
                         nn.Dropout(0.1),
                         nn.Linear(1000, 200),
                         nn.ReLU(),
                         nn.Dropout(0.1),
                         nn.Linear(200, 2),
                         nn.LogSoftmax(dim=1))

### Negative log-likelihood Loss function
criterion = nn.NLLLoss()
### Adam optimzer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)
### reduce on plateau learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, 
                                          cooldown=1, threshold=1e-4, verbose=True)

### transfer model to device (GPU)
model.to(device)
print(model)

### train and validate model
epochs = 20
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    # cc = 0
    for inputs, labels in trainloader:
        # if cc == 3:
        #     break
        # cc += 1
        # plt.imshow(inputs[1,:,:,:].detach().cpu().permute(1,2,0))
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.5f}.. "
                  f"Test loss: {test_loss/len(testloader):.5f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.5f}")
            running_loss = 0
            model.train()
    scheduler.step(test_loss/len(testloader))
    print('\nLr: {}'.format(optimizer.param_groups[0]['lr']))
    
### save model file
torch.save(model, 'model.pth')

### plot train and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


### Inference on test set images 
dest_dir = '/test'
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),])

def predict_image(name):
    im = Image.open(name)
    image_tensor = test_transforms(im).float()
    image_tensor = image_tensor.unsqueeze_(0)
    inputs = Variable(image_tensor)
    inputs = inputs.to(device)
    output = model(inputs)
    index = output.data.cpu().numpy().argmax()
    return index


df1['prediction'] = np.nan 
for n in df1[df1['pav'].isnull()].index:
    try:
        df1.loc[n,'prediction'] = predict_image('..\\test\\' + df1.loc[n,'display_url'])
    except:
        pass