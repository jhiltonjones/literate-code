import torch
import torch.nn as nn
from neural_net import SimpleMLP

# Load
sd = torch.load("simple_mlp.pt", map_location="cpu")  # weights_only=True default is fine now
model = SimpleMLP()
model.load_state_dict(sd)
model.eval()
def predict_and_jacobian(x_value: float):
    x = torch.tensor([[x_value]], dtype=torch.float32, requires_grad=True)
    y = model(x)                                    # (1,1)
    (dy_dx,) = torch.autograd.grad(y, x, torch.ones_like(y))
    return y.item(), dy_dx.item()

y_pred, J = predict_and_jacobian(-3.5)
print(f"Pred: {y_pred:.4f}, dy/dx: {J:.4f}")