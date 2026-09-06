import torch

from torch import nn
from torch.utils.data import DataLoader
from utils import get_device

device = get_device()

class Trainer:
    def __init__(
        self,
        max_epoch = 5,
    ):
        self.max_epoch = max_epoch

    def fit(
        self,
        model,
        data,
        loss_fn,
        batch_size = 128,
        num_workers = 0,
        lr = 0.01,
    ):
        model.train()
        loader = DataLoader(
            dataset = data,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr
        )

        for epoch in range(self.max_epoch):
            running_loss = 0
            data_count = 0

            loss_lst = []
            for bid, (X, y) in enumerate(loader):
                optimizer.zero_grad()
                
                pred = model(X)
                loss = loss_fn(pred, y)
                loss_lst.append(loss.item())

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * y.shape[0]
                data_count += y.shape[0]

            epoch_loss = running_loss / data_count
            # print(running_loss, data_count)
            print(f"Loss for epoch {epoch}: {epoch_loss:.4f}")
