import os
import torch
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Client, CustomDataSet
from utils import load_file, get_device
from models import RNNModel

device = get_device()

parser = argparse.ArgumentParser(description="Baseline training parameters")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--learning_rate", type=float, default=0.00005)
args = parser.parse_args()

# Args
data_path = "../kv_data/kv/"
checkpt_path = "checkpt/"

features = 197

# Hyper Parameters
loss_fn = torch.nn.MSELoss()
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

client_ids = ["0_0","0_1","0_2","0_3","0_4","0_5","1_0","1_1","1_2","1_3","1_4","1_5","2_0","2_1","2_2","2_3","2_4","2_5","3_0","3_1","3_2","3_3","3_4","3_5"]

#test data
train_dataset = torch.utils.data.ConcatDataset([torch.load("trainpt/"+id+".pt") for id in client_ids])
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

#train data
test_dataset = torch.utils.data.ConcatDataset([torch.load("testpt/"+id+".pt") for id in client_ids])
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

#model setup
model = RNNModel(features)
model.train()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in tqdm(range(epochs)):
    print("\n")
    train_batch_loss = []

    for batch_idx, (x, y) in enumerate(train_dataloader):
        outputs = model(x)
        train_loss = loss_fn(outputs, y)
        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
                print('Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(x), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), train_loss.item()))
                
        train_batch_loss.append(train_loss.item())

    train_loss_avg = sum(train_batch_loss)/len(train_batch_loss)
    print('\nTrain loss:', train_loss_avg)

    val_batch_loss = []
    for _, (x,y) in enumerate(test_dataloader):
        outputs = model(x)
        val_loss = loss_fn(outputs,y)
        val_batch_loss.append(val_loss.item())

    val_loss_avg =  sum(val_batch_loss)/len(val_batch_loss)
    print('Validation loss:', val_loss_avg)






#torch.save(model.state_dict(), checkpt_path+"_"+str(batch_size)+"_baseline.pth")