import os
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from utils import Client, CustomDataSet
from utils import load_file, get_device

import torch.optim as optim

from models import ShallowNN

# Args
path = "../kv_data/kv/"
n_clients = 24

files = os.listdir(path)
files_path = [os.path.join(path,file) for file in files]

device = get_device()

uids = [u for u in range(n_clients)]

# Hype Parameters
loss_fn = torch.nn.MSELoss() #nn.MSELoss()
batch_size = 32
features = 197

clients = [Client(i,load_file(files_path[i])) for i in range(len(files_path))][0:1]

epochs=20
local_round_count=10
learning_rate= 0.00005

#initiate global model
global_model = ShallowNN(features)
global_model.to(device)
global_model.train()

# copy weights
global_weights = global_model.state_dict()

model_layers = global_model.track_layers.keys()

""" for key in model_layers:
    global_model.track_layers[key].weight.data = global_weights[str(key) + ".weight"]
"""

# info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2

for epoch in tqdm(range(epochs)):

    local_models , local_loss = [], []

    print(f'\n | Global Training Round : {epoch+1} |\n')

    for client in clients:
        client_id = client.get_client_id
        client.set_model(ShallowNN(features))
        client.get_model().load_state_dict(global_weights)        
        optimizer = torch.optim.SGD(client.get_model().parameters(),lr=learning_rate)
        this_client_state_dict = client.train(client.get_model(), loss_fn, optimizer, batch_size, epoch)
        local_models.append(this_client_state_dict)

    #update global model parameters here


#print(clients[1].get_model().state_dict())
local_models[0]