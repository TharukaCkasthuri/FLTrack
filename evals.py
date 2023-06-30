import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_model(model):
    pass

def test_inference(model, test_dl, loss_fn):
    """

    """
    model.eval()
    loss, mse, mae = 0.0, 0.0, 0.0

    for batch_idx, (x, y) in enumerate(test_dl):
        predicts = model(x)
        batch_loss = loss_fn(predicts,y)
        loss += batch_loss.item()
        
        batch_mse = mean_squared_error(list(y), np.squeeze(predicts.detach().numpy()))
        mse += batch_mse
        batch_mae = mean_absolute_error(list(y), np.squeeze(predicts.detach().numpy()))
        mae += batch_mae

    return loss, mse, mae