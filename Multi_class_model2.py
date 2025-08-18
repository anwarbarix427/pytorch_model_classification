#exercie
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch 
from torch import nn
import matplotlib.pyplot as plt

#create random data
n_samples = 1000
noise = 0.03
random_state = 42
X,y = make_moons(n_samples=n_samples,noise=noise,random_state=random_state)
plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.title("Random Data")
#plt.show()
 
#turn data into tensors 
X=torch.tensor(data=X, dtype=torch.float)
y=torch.tensor(data=y,dtype=torch.long)
print(X[:5])
print(y[:5])

#lets split the data
k=0.2
X_train ,X_test , y_train ,y_test= train_test_split(X,y,test_size=k,random_state=42)
print(f"data size split {len(X_train),len(y_train),len(X_test),len(y_test)}")

#setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using a {device} as a device')
#lets build our model

class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=5)
        self.layer3 = nn.Linear(in_features=5, out_features=2)

    def forward(self, x):
        out = nn.ReLU()(self.layer1(x))
        out = nn.ReLU()(self.layer2(out))
        out = self.layer3(out)
        return out
#made an instance of our model
model = MoonModel()
print(model)

#create a loss function
loss_fn = nn.CrossEntropyLoss()
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    accuracy = (correct / len(y_pred))*100
    return accuracy
optimiser = torch.optim.Adam(params = model.parameters(),lr=0.1)

#creating a training loop and test our model 

torch.manual_seed(seed=42)
X_train,y_train=X_train.to(device),y_train.to(device)
X_test , y_test=X_test.to(device),y_test.to(device)

epochs=200

for epoch in range(epochs):
    model.train()#put the model in training mode
    y_logits = model(X_train)
    y_preds = torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_train)
    accuracy =accuracy_fn(y_preds,y_train)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

#testing phase
    model.eval()
    with torch.inference_mode():
        X_logits = model(X_test)
        y_preds = torch.softmax(X_logits,dim=1).argmax(dim=1)
        loss = loss_fn(X_logits, y_test)
        accuracy = accuracy_fn(y_preds, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Train Loss: {loss.item():.5f}, Train Acc: {accuracy:.2f}%, Test Loss: {loss.item():.5f}, Test Acc: {accuracy:.2f}%")
#ploting our data
from helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()