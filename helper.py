import os
import csv
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



# load data
def load_dataset(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    Wrap data into PyTorch Datasets and DataLoaders.

    Parameters
    ----------
    X_train, X_val, X_test : array-like
        Feature arrays.
    y_train, y_val, y_test : array-like
        Label arrays.
    batch_size : int
        Batch size for DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch DataLoaders for training, validation, and test sets.
    """
    class MyDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

#-------------------------------------------------------------------------------------------------------

# split data
def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):

    """
    Split dataset into training, validation, and test sets.

    Parameters
    ----------
    X : np.ndarray
        Input features.

    y : np.ndarray
        Target variables.

    val_size : float
        Fraction of data to use for validation.

    test_size : float
        Fraction of data to use for testing.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
        Split datasets.
    """


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

#-------------------------------------------------------------------------------------------------------

# training loop

def train_model(model, train_loader, val_loader, config, device):
    """
    Train a PyTorch model with support for CosineAnnealingLR, ReduceLROnPlateau,
    early stopping, optional custom loss, and CSV logging (epoch, train loss, val loss, lr).

    Parameters
    ----------
    model : torch.nn.Module
        Neural network to train.
    train_loader, val_loader : torch.utils.data.DataLoader
        DataLoaders for training and validation.
    config : dict
        Hyperparameters and training options. Keys:
            - "epochs" (int)
            - "lr" (float)
            - "optimizer" (str) "adam" or "sgd"
            - "weight_decay" (float, optional)
            - "loss_fn" (optional, callable, default MSE)
            - "cosine" (dict or None): {"T_max": int}
            - "plateau" (dict or None): {"patience": int, "factor": float}
            - "early_stop" (dict or None): {"patience": int}
            - "save_path" (str, optional): CSV log path
    device : torch.device
        Device to run training on.

    Returns
    -------
    train_losses, val_losses : list of float
        Average training and validation losses per epoch.
    """
    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)
    save_path = config.get("save_path", "train_log.csv")

    # Optimizer
    if config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    # Loss function
    loss_fn = config.get("loss_fn", torch.nn.MSELoss())

    # Scheduler
    scheduler = None
    scheduler_type = None
    if config.get("cosine"):
        T_max = config["cosine"].get("T_max", epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        scheduler_type = "cosine"
    elif config.get("plateau"):
        factor = config["plateau"].get("factor", 0.1)
        patience = config["plateau"].get("patience", 5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience)
        scheduler_type = "plateau"

    # Early stopping
    early_stop_patience = None
    best_val_loss = float("inf")
    epochs_no_improve = 0
    if config.get("early_stop"):
        early_stop_patience = config["early_stop"].get("patience", 10)

    model.to(device)
    train_losses, val_losses = [], []

    # CSV init
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += loss_fn(preds, y_batch).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # CSV log
        with open(save_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, current_lr])

        # Print progress
        if epoch%10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Scheduler step
        if scheduler:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:  # cosine
                scheduler.step()

        # Early stopping
        if early_stop_patience:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return train_losses, val_losses

#-------------------------------------------------------------------------------------------------------

# testing 

def evaluate_model(model, test_loader, device, loss_fn=torch.nn.MSELoss(), precomputed = False):
    """
    Evaluate a PyTorch model on a test set and return predictions and true labels.
    Also computes and prints the average test loss.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set.
    device : torch.device
        Device to run evaluation on.
    loss_fn : callable, optional
        Loss function to compute test loss (default: MSELoss).
    precomputed : bool
        Uses precomputed weights if `best_model.pth` exists

    Returns
    -------
    preds : np.ndarray
        Model predictions on the test set.
    y_true : np.ndarray
        True labels from the test set.
    """
    # Load saved weights if requested
    if precomputed:
        if os.path.exists("best_model.pth"):
            checkpoint = torch.load("best_model.pth", map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            print("Loaded precomputed weights from best_model.pth")
        else:
            print("Warning: best_model.pth not found")
    
    model.to(device)

    model.eval()
    preds_list = []
    y_true_list = []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            
            # Compute batch loss
            batch_loss = loss_fn(preds, y_batch).item()
            total_loss += batch_loss * X_batch.size(0)  # multiply by batch size

            preds_list.append(preds.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    # Average loss over all samples
    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Average test loss: {avg_loss:.4f}")

    return np.vstack(preds_list), np.vstack(y_true_list)

#-------------------------------------------------------------------------------------------------------

# plot loss curve

def plot_loss(log_path="train_log.csv", log_scale=False):
    """
    Plot training and validation loss curves from a CSV log file.

    Parameters
    ----------
    log_path : str
        Path to the CSV file containing training logs. Defaults to "train_log.csv"

    log_scale : bool, optional
        If True, use logarithmic scale for the x-axis.
    """

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found")

    # Load CSV
    df = pd.read_csv(log_path)

    # Extract columns
    epochs = df["epoch"].values
    train_losses = df["train_loss"].values
    val_losses = df["val_loss"].values

    # Plot
    plt.figure()

    if log_scale:
        plt.xscale("log")

    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()


#-------------------------------------------------------------------------------------------------------

# plot residuals

def plot_residuals(y_true, y_pred, label_names):
    """
    Plot residuals (errors) as a function of predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (N, D).

    y_pred : np.ndarray
        Predicted values (N, D).

    label_names : list of str
        Names of each target dimension.

    """
    residuals = y_true - y_pred

    for i, label in enumerate(label_names):
        plt.figure()
        plt.hist(residuals[:, i], bins=30, alpha=0.7)
        plt.axvline(0, color='r', linestyle='--', label = "Ideal")
        plt.xlabel("Residual")
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Residual distribution: {label}")
        plt.show()

#-------------------------------------------------------------------------------------------------------

# plot true v/s predicted

def plot_true_vs_pred(y_true, y_pred, label_names, alpha = 0.5):
    """
    Generate scatter plots comparing true vs predicted values for each target.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (N, D).

    y_pred : np.ndarray
        Predicted values (N, D).

    label_names : list of str
        Names of each target dimension (length D).

    alpha : float
        Transparency of scatter plot


    """
    for i in range(len(label_names)):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=alpha)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs True values for {label_names[i]}")
        plt.plot(
            [y_true[:, i].min(), y_true[:, i].max()],
            [y_true[:, i].min(), y_true[:, i].max()],
            'r--',
            linewidth=2,
            label = "Ground Truth"
        )
        plt.show()

#-------------------------------------------------------------------------------------------------------

# plot 2D heatmap

def plot_heatmap(y_true, y_pred, label_names):
    """
    Plot a 2D histogram (heatmap) comparing two dimensions of true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (N, D).

    y_pred : np.ndarray
        Predicted values (N, D).

    label_names : list of str
        Names of the labels

    """

    for i in range(len(label_names)):
        plt.hist2d(y_true[:, i], y_pred[:, i], bins=50, cmap = "magma")
        plt.colorbar(label = "Density")
        plt.xlabel(f"True value")
        plt.ylabel(f"Predicted value")
        plt.title(f"Bivariate Density Plot for {label_names[i]}")
        plt.show()

#-------------------------------------------------------------------------------------------------------

# normalize after splitting
def normalize_labels(y_train, y_val, y_test):
    """
    Normalize target variables using statistics computed from the training set.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels of shape (N_train, D).

    y_val : np.ndarray
        Validation labels of shape (N_val, D).

    y_test : np.ndarray
        Test labels of shape (N_test, D).

    Returns
    -------
    y_train_norm : np.ndarray
        Normalized training labels.

    y_val_norm : np.ndarray
        Normalized validation labels.

    y_test_norm : np.ndarray
        Normalized test labels.

    mean : np.ndarray
        Mean of training labels (shape: D,).

    std : np.ndarray
        Standard deviation of training labels (shape: D,).

    """

    mean = y_train.mean(axis=0)
    std = y_train.std(axis=0)

    y_train_norm = (y_train - mean) / std
    y_val_norm   = (y_val   - mean) / std
    y_test_norm  = (y_test  - mean) / std

    return y_train_norm, y_val_norm, y_test_norm, mean, std

#-------------------------------------------------------------------------------------------------------

# denormalize
def denormalize_labels(y_norm, mean, std):
    """
    Reverse normalization to recover original scale of target variables.

    Parameters
    ----------
    y_norm : np.ndarray
        Normalized data.

    mean : np.ndarray
        Mean used during normalization.

    std : np.ndarray
        Standard deviation used during normalization.

    Returns
    -------
    y : np.ndarray
        Data transformed back to original scale.
    """

    return y_norm * std + mean

