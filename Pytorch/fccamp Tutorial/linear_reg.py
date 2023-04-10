import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class linearRegresion(nn.Module):
    def __init__ (self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights*x + self.bias
    




if __name__ == "__main__":
    # y = w*X + b --> X es el tensor con los "features" e y son los "labels"
    w = 0.7
    b = 0.3
    start = 0
    end = 1
    step = 0.02

    # Armamos el dataset
    X = torch.arange(start, end, step)
    y = w*X + b

    #Test y training dataset slice
    train_slice = int(0.8*len(X))

    X_train = X[:train_slice]
    y_train = y[:train_slice]

    X_test = X[train_slice:]
    y_test =  y[train_slice:]
    
    def plotPredictions(data_train = X_train, labels_train = y_train, data_test = X_test, labels_test = y_test, predictions = None, title = "no title"):
        plt.figure()
        #plot training data
        plt.scatter(data_train, labels_train, c= "b", s = 4, label = "Training Dataset")

        #plot test data
        plt.scatter(data_test, labels_test, c="g", s=4, label = "Test Dataset")

        if predictions !=None:
            plt.scatter(data_test, predictions, c = "r", s = 4, label = "Predictions")
        plt.legend(prop = {"size": 14})
        plt.title(title)
        plt.show()
        return None
    plotPredictions(title =  "Dataset")

    #Creamos el modelo

    torch.manual_seed(42)
    modelo = linearRegresion()
    print(f"Estado del modelo: {modelo.state_dict()}")

    #predicciones del modelo sin entrenar

    with torch.inference_mode():
        no_trained_preds = modelo (X_test)

    plotPredictions(predictions=no_trained_preds, title = "No Trained")

    epochs = 10000
    loss_function = nn.L1Loss()
    optimizer = torch.optim.SGD(params=modelo.parameters(), lr = 0.01)
    torch.manual_seed(42)
    epochs_data = []
    train_loss_data = []
    test_loss_data = []
    for epoch in range(epochs):
        # Train mode
        modelo.train()

        #Forward pass
        predictions = modelo(X_train)

        #Calulte loss
        loss = loss_function(predictions, y_train)

        #Reset otimizer grads
        optimizer.zero_grad()

        #backprop
        loss.backward()

        #step optimizer

        optimizer.step()

        #vamos probando el modelo a medida que lo entrenamos
        modelo.eval()
        with torch.inference_mode():
            test_pred = modelo(X_test)
            test_loss = loss_function(test_pred, y_test)
        
        epochs_data.append(epoch)
        train_loss_data.append(loss.detach().numpy())
        test_loss_data.append(test_loss.detach().numpy())
    
    with torch.inference_mode():
        trained_preds = modelo(X_test)
    plotPredictions(predictions=trained_preds, title = "Trained")

    plt.figure()
    plt.plot(epochs_data, train_loss_data , label = "Train Loss", c = "r")
    plt.plot(epochs_data, test_loss_data, label = "Test loss", c = "b")
    plt.legend()
    plt.show()
    with torch.inference_mode():
        trained_preds = modelo(X_test)

    print("algo")








