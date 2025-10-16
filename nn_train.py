import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from neural_net import SimpleMLP
from formatted_data import data
import pandas as pd

df = pd.DataFrame(data, columns=["Joint_6", "Beam_Angle_Deg"])
X_np = df[["Joint_6"]].values
y_np = df[["Beam_Angle_Deg"]].values
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype = torch.float32)
model = SimpleMLP()
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(5000):
    opt.zero_grad()
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    loss.backward()
    opt.step()
    if (epoch+1)%200 ==0:
        print(f"epoch{epoch+1:4d} loss {loss.item():.6f}")
torch.save(model.state_dict(), "simple_mlp.pt")  # note the parentheses!
