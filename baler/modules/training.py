import sys
import time
from torch import optim
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import modules.utils as utils


def fit(model, train_dl, train_ds, model_children, regular_param, optimizer, RHO, l1):
    print("Training")
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(
        enumerate(train_dl), total=int(len(train_ds) / train_dl.batch_size)
    ):
        counter += 1
        x, _ = data
        optimizer.zero_grad()
        reconstructions = model(x)

        if l1:
            loss = utils.Georges_sparse_loss_function_L1(
                model_children=model_children,
                true_data=x,
                reconstructed_data=reconstructions,
                reg_param=regular_param,
            )
        else:
            loss = utils.sparse_loss_function_KL(
                rho=RHO,
                model_children=model_children,
                true_data=x,
                reconstructed_data=reconstructions,
                reg_param=regular_param,
            )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f" Train Loss: {loss:.6f}")

    return epoch_loss, model


def validate(model, test_dl, test_ds, model_children, reg_param):
    print("Validating")
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(test_dl), total=int(len(test_ds) / test_dl.batch_size)
        ):
            counter += 1
            x, _ = data
            reconstructions = model(x)
            loss = utils.Georges_sparse_loss_function_L1(
                model_children=model_children,
                true_data=x,
                reconstructed_data=reconstructions,
                evaluate=False,
                reg_param=reg_param,
            )
            running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f" Val Loss: {loss:.6f}")
    # save the reconstructed images every 5 epochs
    return epoch_loss


def train(model, input_dim, train_data, test_data, project_path, config, device):
    epochs = config["epochs"]
    reg_param = config["reg_param"]
    l1 = config["l1"]
    z = config["latent_space_size"]
    bs = config["batch_size"]
    learning_rate = config["lr"]
    RHO = config["RHO"]
    sae = model
    model_children = list(sae.children())

    # Constructs a tensor object of the data and wraps them in a TensorDataset object.

    train_ds = TensorDataset(
        torch.tensor(train_data.values, dtype=torch.float64),
        torch.tensor(train_data.values, dtype=torch.float64),
    )
    valid_ds = TensorDataset(
        torch.tensor(test_data.values, dtype=torch.float64),
        torch.tensor(test_data.values, dtype=torch.float64),
    )

    # Converts the TensorDataset into a DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    ## Activate early stopping
    if config["early_stopping"] == True:
        early_stopping = utils.EarlyStopping(
            patience=config["patience"], min_delta=config["min_delta"]
        )  # Changes to patience & min_delta can be made in configs

    if config["lr_scheduler"] == True:
        lr_scheduler = utils.LRScheduler(
            optimizer=optimizer, patience=config["patience"]
        )

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, trained_model = fit(
            model=sae,
            train_dl=train_dl,
            train_ds=train_ds,
            model_children=model_children,
            optimizer=optimizer,
            RHO=RHO,
            regular_param=reg_param,
            l1=l1,
        )
        val_epoch_loss = validate(
            model=trained_model,
            test_dl=valid_dl,
            test_ds=valid_ds,
            model_children=model_children,
            reg_param=reg_param,
        )
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        if config["lr_scheduler"] == True:
            lr_scheduler(train_epoch_loss)

        if config["early_stopping"] == True:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break
    end = time.time()

    pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss}).to_csv(
        project_path + "loss_data.csv"
    )

    print(f"{(end - start) / 60:.3} minutes")

    data = torch.tensor(test_data.values, dtype=torch.float64)

    pred = trained_model(data)
    pred = pred.detach().numpy()
    data = data.detach().numpy()

    return data, pred
