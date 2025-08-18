import torch
from torch import nn
from pathlib import Path
weight =0.7
bias = 0.3
X = torch.arange(0,1,0.02).unsqueeze(dim=1)
Y = weight * X + bias
print(X,Y,)
Train_split = 0.8*len(X)

#preparing the data we are going to use
X_train = X[:int(Train_split)]
Y_train = Y[:int(Train_split)]
X_test = X[int(Train_split):]
Y_test = Y[int(Train_split):]


class LineaireRegressionSimple(nn.Module):
    def __init__(self):
        super(LineaireRegressionSimple, self).__init__()
        self.poids = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)

    def forward(self, x):
        return self.poids * x + self.bias

#Create an instance of our class and check about paremeters and what is its contenu

model = LineaireRegressionSimple()
print(list(model.parameters()))
print(model.state_dict())


#we are going to use .inference_mode() to make prediction with ouir builded model

with torch.inference_mode():
    y_pred = model(X_test)


#Create a loss function and an optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#creating a training and testing loop
torch.manual_seed(42)
epochs = 500

#create an empty loss list to tracks loss values
Train_loss = []
Test_loss = []
epochs_count=[]

for epoch in range(epochs):
    #Training
    model.train()
    y_pred = model(X_train)
    #calculate the loss
    loss = loss_fn(y_pred, Y_train)
    
    #optimizer zero grad
    optimizer.zero_grad()
    
    #backward pass
    loss.backward()
    
    #optimizer step
    optimizer.step()
    
    #append the loss to the list
    Train_loss.append(loss.item())
    
    #Testing
    model.eval()
    with torch.inference_mode():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred, Y_test)
        Test_loss.append(test_loss.item())
    
    epochs_count.append(epoch)
print("Training complete.\n")
print("the model learned those values")
print(f"Weight: {model.poids.item()}, Bias: {model.bias.item()}\n")
print(f"the original values are {weight}, {bias}\n")


#save and load our model
path_model=Path('models')
path_model.mkdir(parents=True, exist_ok=True)
model_save_path=path_model / "linear_regression_model.pth"
print(f'saving model to {model_save_path}')
torch.save(obj=model.state_dict(), f=model_save_path)