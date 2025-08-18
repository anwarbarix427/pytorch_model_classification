#multi_class data classification model
#create a multi_class data using sklearn.datasets 

import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


n_samples = 1000
n_classes = 4
n_features = 2
X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features, random_state=42)


#turn data into tensors

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)  # Use long for class labels

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 #ploting the results(data)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k', s=50)
plt.title("Multi-Class Data Distribution")
plt.show()

#configur the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device\n")

#build the model
class BlobModel(nn.Module):
    def __init__(self, input_features, hidden_units, output_classes):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            #nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_classes)  # Output layer for multi-class classification
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

#Create an instance of BlobModel and send it to the target device
model = BlobModel(input_features=n_features, hidden_units=10, output_classes=n_classes).to(device)
print(model)

# Creating a loss function and optimizer for a multi-class PyTorch model
loss_fn = nn.CrossEntropyLoss()
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct / len(y_pred))*100
    return accuracy
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)



#make predictions
y_logits = model(X_test.to(device))
y_pred_probs = torch.softmax(y_logits, dim=1)
#print(y_logits[:5])
#print(y_pred_probs[:5])

# Creating a training and testing loop for a multi-class PyTorch model

torch.manual_seed(seed=42)
epochs = 100
X_train , y_train = X_train.to(device),y_train.to(device)
X_test , y_test  = X_test.to(device),y_test.to(device)

for epoch in range(epochs):
    #put the trainig mode 
    model.train()
    #forward pass
    y_logits = model(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    #calculate the loss and the accuarcy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #testing
    model.eval()
    with torch.inference_mode():
        x_logits = model(X_test)
        y_pred = torch.softmax(x_logits, dim=1).argmax(dim=1)
        loss_test = loss_fn(x_logits, y_test)
        acc_test = accuracy_fn(y_test, y_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Train Loss: {loss.item():.5f}, Train Acc: {acc:.2f}%, Test Loss: {loss_test.item():.5f}, Test Acc: {acc_test:.2f}%")
