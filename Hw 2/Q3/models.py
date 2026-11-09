"""
Skeleton code for creating the Random Kitchen Sinks (RKS) and 1-hidden-layer MLP (MLP) models.

Note you are *welcome* to completely ignore this file and use your own code, but it's here if you want to use it as a starting point.
"""
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn

"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            D: total number of random features (= 2h in the assignment text; if you use both sin and cos features, D should be an even number)
            sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
            ridge_lambda: ridge regularization strength
        """
        # TODO: initialize the model parameters
        self.D = D
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda


    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the input data X into the random kitchen sinks features. You will likely need to use the parameters initialized in the __init__ method.
        The resulting features should be of shape (n, D or D[+1]), where the plus one is for the intercept term.
        """
        # TODO: implement the feature transformation
        # Calculate random projections
        projection = X @ self.W + self.b

        # Generate sine and cosine features
        Z_sin = np.sin(projection)
        Z_cos = np.cos(projection)

        # Add intercept column
        Z = np.concatenate([Z_sin, Z_cos], axis=1)
        Z = np.column_stack([np.ones(X.shape[0]), Z])

        return Z


        

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        # TODO: optional code before getting features
        # Number of sine/cosine feature pairs
        h = self.D // 2

        # Initialize random weights and offsets
        self.W = np.random.normal(
            loc=0,
            scale=1 / self.sigma,
            size=(X.shape[1], h)
        )

        self.b = np.random.uniform(
            low=0,
            high=2 * np.pi,
            size=h
        )

        phi = self._features(X)  # shape (n, D or D[+1])
        # TODO: fit the ridge regression model and obtain the fitted ridge coefficients
        # Ridge regression with an unregularized intercept
        penalty = np.eye(self.D + 1)
        penalty[0, 0] = 0

        self.coef_ = np.linalg.solve(
            phi.T @ phi + self.ridge_lambda * penalty,
            phi.T @ y
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # shape (n, D or D[+1])
        return phi @ self.coef_  # shape (n,)

def sweep_rks_model(sweep_D: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    An example sweeping function that sweeps over the number of random features D, it then returns the best model in terms of validation MSE.
    Args:
        sweep_D: list of numbers of random features to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
    Returns:
        a tuple (best_model, best_val_mse, best_D)
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None

    train_losses = []
    val_losses = []

    for D in sweep_D:
        # TODO: fit the model and get the validation MSE
        model = RandomKitchenSinks(D=D, sigma=1.0, ridge_lambda=1e-3)
        model.fit(X_train, y_train)

        # Calculate predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Calculate MSE
        train_mse = np.mean((y_train - train_pred) ** 2)
        val_mse = np.mean((y_val - val_pred) ** 2)

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        print(
            f"D = {D}: "
            f"Train MSE = {train_mse:.4f}, "
            f"Validation MSE = {val_mse:.4f}"
        )

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_D = D

    print(
        f"\nBest hyperparameters: D = {best_D}, "
        f"sigma = 1.0, ridge_lambda = 1e-3"
    )
    print(f"Best validation MSE: {best_val_mse:.4f}")

    return best_model, best_val_mse, best_D, train_losses, val_losses

"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128, activation: str = "relu", device: str = "cpu"):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            d_hidden: hidden dimension
            activation: activation function
            lr: learning rate
            batch_size: batch size
        """
        super().__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation

        # DO NOT forget this -- GPU will make NN training much faster if you have access to one
        self.device = device
        # TODO: build up the 1-hidden-layer MLP model with modules like `nn.Linear`, `nn.Sequential`, `nn.ReLU`, etc.
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

        self.net.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # turn the numpy data into torch tensors and move to the device
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        # Create dataset and dataloader
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.net.train()  # make sure to set the model to training mode
        # TODO: fit the MLP model and obtain the fitted MLP coefficients
        # optimizer = xxx
        # loss_fn = yyy
        # for epoch in range(num_epochs):
        # set up batch data loader and fit the model within each batch

        # Initialize optimizer and loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                pred_train = self.forward(X_batch)

                # Calculate loss
                loss = loss_fn(y_batch, pred_train)

                # Update weights
                loss.backward()
                optimizer.step()

        return self


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy()

def mlp_sweep(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    See the example `sweep_rks_model` above for inspiration. Good luck!
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None

    train_losses = []
    val_losses = []
    
    for d_hidden in sweep_values:
        # Initialize and fit model
        model = MLP(
            d_in = X_train.shape[1], 
            d_hidden = d_hidden, 
            d_out = 1,
            lr = 1e-3, 
            batch_size = 128, 
            activation = "relu", 
            device = "xpu"
        )
        model.fit(X_train, y_train)

        # Get predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Calculate MSE
        train_mse = np.mean((y_train - train_pred) ** 2)
        val_mse = np.mean((y_val - val_pred) ** 2)

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        print(
            f"Hidden Dim = {d_hidden}: "
            f"Train MSE = {train_mse:.4f}, "
            f"Validation MSE = {val_mse:.4f}"
        )

        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_d_hidden = d_hidden

    print(
        f"\nBest hyperparameters: Hidden Dim = {best_d_hidden}, "
        f"learning rate = 1e-3, batch size = 128, activation = relu"
    )
    print(f"Best validation MSE: {best_val_mse:.4f}")

    return best_model, best_val_mse, best_d_hidden, train_losses, val_losses
