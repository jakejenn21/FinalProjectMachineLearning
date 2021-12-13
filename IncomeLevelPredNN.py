import csv
import pprint
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from torch import nn
from sklearn.preprocessing import OrdinalEncoder


class NN(nn.Module):
    def __init__(self, hidden_width, depth, activation):
        super(NN, self).__init__()
        layers = []
        input_layer = nn.Linear(14, hidden_width)
        if activation == "relu":
            activation_input = nn.ReLU()
        else:
            activation_input = nn.Tanh()

        layers.append(input_layer)
        layers.append(activation_input)
        for i in range(depth-1):
            hidden_layer = nn.Linear(hidden_width, hidden_width)
            if activation == "relu":
                activation_hidden = nn.ReLU()
            else:
                activation_hidden = nn.Tanh()

            layers.append(hidden_layer)
            layers.append(activation_hidden)

        output_layer = nn.Linear(hidden_width, 1)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

        def init_weights_relu(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)

        def init_weights_tanh(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

        if activation == 'relu':
            self.model.apply(init_weights_relu)
        else:
            self.model.apply(init_weights_tanh)

    def forward(self, X):
        input = np.float32(X)
        out = torch.from_numpy(input)
        out.requires_grad = True
        return self.model(out)


# prepare input data
def prepare_inputs(X_train, X_test):

    enc = OrdinalEncoder()
    traindfOE = enc.fit_transform(X_train.astype(str))
    testdfOE = enc.fit_transform(X_test.astype(str))
    return traindfOE, testdfOE


def replace_missing(data):
    for col in data.columns:
        majority_class_index = np.argmax(
            np.unique(data[col], return_counts=True)[1])
        majority_value = np.unique(data[col])[majority_class_index]
        if majority_value == "?":
          newdata = data[data[col] != "?"]
          majority_class_index = np.argmax(
              np.unique(newdata[col], return_counts=True)[1])
          majority_value = np.unique(newdata[col])[majority_class_index]

        data[col] = np.where(data[col] == "?", majority_value, data[col])


# X Train
traindf = pd.read_csv("train_final.csv")

#replace_missing(traindf)
X_train = traindf.iloc[:, :-1]

# Y Train
y_train = traindf.iloc[:, -1]

#y_train = np.where(y_train == 1, 1.0, -1.0)

# X Test
testdf = pd.read_csv("test_final.csv")

#replace_missing(testdf)
X_test = testdf.iloc[:, 1:]

# preprocessing
X_train, X_test = prepare_inputs(X_train, X_test)

print(X_train)
print(X_test)
print(y_train)

criterion = torch.nn.MSELoss(reduction='sum')

# depth = [3, 5, 9]
# width = [5, 10, 25, 50, 100]


width = 10
depth = 5


print("\nTensorFlow Test/Train errors based on Width/Depth\n\n")

print("Tanh:\n")

NeuralNet = NN(width, depth, "tanh")
optimizer = torch.optim.Adam(NeuralNet.parameters())
for t in range(200):

    optimizer.zero_grad()

    # Forward pass: compute predicted y by passing x to the model.
    y_train_pred_tensor = NeuralNet.forward(X_train).flatten()

    y_train_np = np.float32(y_train.copy())
    y_train_tensor = torch.from_numpy(y_train_np)
    y_train_tensor.requires_grad = True
    y_train_tensor = y_train_pred_tensor.flatten()

    print(y_train_pred_tensor)

    print(y_train_tensor)

    loss = criterion(y_train_pred_tensor, y_train_tensor)

    # Backward pass: compute gradient of the loss with respect to model
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()

    test_pred = NeuralNet.forward(X_test.values)

    print(test_pred)

    test_pred[test_pred >= 0] = 1

    test_pred[test_pred < 0] = 0

    test_pred = test_pred.detach().numpy().flatten()

rows = []
for i in testdf.index:
    row = tuple((testdf['ID'][i], test_pred[i]))
    rows.append(row)

outdf = pd.DataFrame(rows, columns=["ID","Prediction"])
outdf = outdf.set_index('ID')
outdf.to_csv("out-" + "width-" + str(width) + "-depth-" + str(depth) + ".csv")

