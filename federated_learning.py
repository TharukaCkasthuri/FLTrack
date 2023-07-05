import os
import torch
import argparse

from tqdm import tqdm

from utils import Client
from utils import load_file, get_device

from models import ShallowNN

from torch.utils.tensorboard import SummaryWriter

device = get_device()

parser = argparse.ArgumentParser(description="Federated training parameters")
parser.add_argument("--batch_size",type=int, default=128)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--learning_rate",type=float, default=0.00005)
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

writer = SummaryWriter(comment="_federated_training_batch_size_"+str(batch_size))

files = os.listdir(data_path)
files_path = [os.path.join(data_path, file) for file in files]
clients = [Client(i+1, load_file(files_path[i])) for i in range(len(files_path))]


# initiate global model
global_model = ShallowNN(features)
#global_model.to(device)
global_model.train()
global_weights = global_model.state_dict()
model_layers = global_model.track_layers.keys()


for epoch in tqdm(range(epochs)):

    local_models, local_loss = [], []

    print(f'\n | Global Training Round : {epoch+1} |\n')

    for client in clients:

        client_id = client.get_client_id()
        client.set_model(ShallowNN(features))

        # loading the global model weights to client model
        client.get_model().load_state_dict(global_weights)

        # setting up the optimizer
        optimizer = torch.optim.SGD(
            client.get_model().parameters(), lr=learning_rate)
        
        # training
        this_client_state_dict, client_loss = client.train(
            client.get_model(), loss_fn, optimizer, batch_size, epoch)
        local_models.append(this_client_state_dict)
        
        writer.add_scalar("Client_"+str(client_id) +
                          " Training Loss", client_loss, epoch)
        
        
    # update global model parameters here
    state_dicts = [model.state_dict() for model in local_models]
    for key in model_layers:
        global_model.track_layers[key].weight.data = torch.stack([item[str(key)+ ".weight"] for item in state_dicts]).mean(dim=0)
        global_model.track_layers[key].bias.data = torch.stack([item[str(key)+ ".bias"] for item in state_dicts]).mean(dim=0)
        # info here - https://discuss.pytorch.org/t/how-to-change-weights-and-bias-nn-module-layers/93065/2


    global_weights = global_model.state_dict()

writer.flush()
writer.close()

global_model.eval()
torch.save(global_model.state_dict(), checkpt_path+"_"+str(batch_size)+"_fedl_global.pth")
