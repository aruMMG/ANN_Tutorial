# A tiny ANN in PyTorch (XOR) — clean + beginner friendly
import torch
import torch.nn as nn
from model import NN

torch.manual_seed(0)
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])
y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])


lr = 0.5
epochs = 300

model = NN(2, 4, 1)
loss_fn = nn.BCELoss()  # binary cross-entropy (expects outputs in [0,1])
opt = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    y_hat = model(X)    #   Forward Pass
    loss = loss_fn(y_hat, y)    #   Compute loss
    loss.backward() #   Back-propagation (gradient calculation)
    # opt.step()  # Gradient descent step
    with torch.no_grad():  
        for p in model.parameters():
            p -= lr * p.grad
    opt.zero_grad()


    model.zero_grad()
    if epoch % 10 == 0:
        print(f"epoch {epoch:4d} | loss = {loss.item():.4f}")

# -------------------------
# 5) Check predictions
# -------------------------
with torch.no_grad():
    probs = model(X)
    preds = (probs > 0.5).float()

print("\nX:\n", X)
print("probabilities:\n", probs)
print("predictions:\n", preds)
print("targets:\n", y)
# ...


