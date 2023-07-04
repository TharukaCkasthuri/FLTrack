import os
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Client, CustomDataSet
from utils import load_file, get_device
from models import ShallowNN
from evals import test_inference

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = get_device()

# Args
data_path = "../kv_data/kv/"
checkpt_path = "checkpt/"
epochs = 120

# Hyper Parameters
loss_fn = torch.nn.MSELoss() #nn.MSELoss()
batch_size = 128
features = 197
learning_rate= 0.00005

#init data
files = os.listdir(data_path)
files_path = [os.path.join(data_path,file) for file in files]
clients = [Client(i,load_file(files_path[i])) for i in range(len(files_path))]

#train data
x_train = pd.concat([client.get_x_train() for client in clients], axis=0,  ignore_index=True)
y_train = np.concatenate(tuple([client.get_y_train() for client in clients]))
train_dataset = CustomDataSet(x_train,y_train)
trainloader = DataLoader(train_dataset,batch_size,shuffle=True)

#test data
x_test = pd.concat([client.get_x_test() for client in clients], axis=0,  ignore_index=True)
y_test = np.concatenate(tuple([client.get_y_test() for client in clients]))
test_dataset = CustomDataSet(x_test,y_test)
testloader = DataLoader(test_dataset,batch_size,shuffle=True)

#model setup
model = ShallowNN(features)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

epoch_loss = []

for epoch in tqdm(range(epochs)):
    batch_loss = []

    for batch_idx, (x, y) in enumerate(trainloader):
        outputs = model(x)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
                print('Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(x), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
                
        batch_loss.append(loss.item())

    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain loss:', loss_avg)
    writer.add_scalar("Baseline Train Loss", loss_avg, epoch)
    epoch_loss.append(loss_avg)

writer.flush()
writer.close()

model.eval()
torch.save(model.state_dict(), checkpt_path+"baseline.pth")

loss, mse, mae = test_inference(model,testloader,loss_fn)
print(loss)
print(mse)
print(mae)