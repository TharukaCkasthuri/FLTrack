import os
import torch
import argparse
from tqdm import tqdm

from utils import Client
from utils import get_device

from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter

device = get_device()

parser = argparse.ArgumentParser(description="Isolated client training parameters")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int,  default=1000)
parser.add_argument("--learning_rate", type=float, default=0.01)
args = parser.parse_args()

# Args
data_path = "../kv_data/kv/"
checkpt_path = "checkpt/isolated/"

features = 197

# Hyper Parameters
loss_fn = torch.nn.L1Loss() 
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

writer = SummaryWriter(comment="_isolated_training_batch_size_"+str(batch_size))

client_ids = ["0_0","0_1","0_2","0_3","0_4","0_5","1_0","1_1","1_2","1_3","1_4","1_5","2_0","2_1","2_2","2_3","2_4","2_5","3_0","3_1","3_2","3_3","3_4","3_5"]

clients = [Client(id, torch.load("trainpt/"+id+".pt"), torch.load("testpt/"+id+".pt"), batch_size) for id in client_ids]

for client in clients:
    client_id = client.client_id
    client_model = ShallowNN(features)
    
    optimizer = torch.optim.SGD(
            client_model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(epochs)):

        client_model , train_loss = client.train(
            client_model, loss_fn, optimizer, epoch)
        
        validation_loss = client.eval(client_model, loss_fn)
        print('Train loss:', train_loss)
        print('Validation loss:', validation_loss,"\n")

        if epoch >= 5:
            writer.add_scalars("Client_"+str(client_id) +
                          " Loss", {"Training Loss":train_loss, "Validation Loss": validation_loss}, epoch)
        
        
    model_path =  checkpt_path + "client_" + str(client_id) +".pth"
    saving_model = client_model.eval()
    torch.save(saving_model.state_dict(), model_path)
