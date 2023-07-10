import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_model(model):
    pass

def test_inference(model, test_dl, loss_fn):
    """

    """
    model.eval()
    loss, mse, mae = [], [], []

    for batch_idx, (x, y) in enumerate(test_dl):
        predicts = model(x)
        batch_loss = loss_fn(predicts,y)
        loss.append(batch_loss.item())
        
        batch_mse = mean_squared_error(list(y), np.squeeze(predicts.detach().numpy()))
        mse.append(batch_mse)
        batch_mae = mean_absolute_error(list(y), np.squeeze(predicts.detach().numpy()))
        mae.append(batch_mae)

    print(max(loss))
    print(min(loss))
    return sum(loss)/len(loss), sum(mse)/len(mse)