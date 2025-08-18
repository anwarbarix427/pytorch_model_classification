#Pytorch Neural Network Classification

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

#create a dataset to have a probleme of colassification
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)
#show the first five(5) points cordination and the  first five(5) labels (1 or 0)
print(f"First five points:\n{X[:5]}\n")
print(f"First five labels:\n{y[:5]}\n")


# make dataframe for circle using pandas
circles = pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'label':y[:]})
print(circles.head())
print(circles.label.value_counts())

plt.scatter(circles.x1, y=circles.x2, c=circles.label, cmap=plt.cm.RdYlBu)
plt.show()

#turn data into tensors

X=torch.tensor(X, dtype=torch.float)
y=torch.tensor(y, dtype=torch.float)
print(f"X shape: {X.shape}, y shape: {y.shape},\n")
#print(X,y)


# Split data into train and test sets

from sklearn.model_selection import train_test_split
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#print(len(X_train), len(X_test), len(y_train), len(y_test))

#let's try to build a model 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

class CircleModel0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

#create an instance of the model
model_0 = CircleModel0().to(device)
print(model_0)
 

 #make preds
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

#setup a loss function and an optimizer
loss_fn=torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr=0.1)
#lets create an accuarcy function to measure how much our model is going right

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct / len(y_pred))*100
    return accuracy

#let's Train our model
torch.manual_seed(seed=42)
epochs = 100
#put the data into pur device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))
# calculate the loss
loss = loss_fn(y_logits, y_train)
accuracy = accuracy_fn(y_train, y_preds)
#set the optimzer to zero
optimizer.zero_grad()
#backward pass
loss.backward()
#step the optimizer
optimizer.step()
model_0.eval()
with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
if epoch % 10 == 0:
    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}%")