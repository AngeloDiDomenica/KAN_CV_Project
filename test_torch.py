import torch

# crea tensor
x = torch.tensor([1.0, 2.0, 3.0])

print("Tensor:")
print(x)

# usa GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice usato:")
print(device)

# sposta tensor su GPU
x = x.to(device)

print("\nTensor sulla GPU:")
print(x)