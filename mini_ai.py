import torch
import torch.nn as nn
import torch.optim as optim

# dati di training
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).cuda()
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]]).cuda()

# modello semplice
model = nn.Linear(1, 1).cuda()

# funzione errore
criterion = nn.MSELoss()

# ottimizzatore
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training
for epoch in range(1000):

    # forward
    predictions = model(x)

    # calcolo errore
    loss = criterion(predictions, y)

    # reset gradienti
    optimizer.zero_grad()

    # backward
    loss.backward()

    # update pesi
    optimizer.step()

# test finale
test = torch.tensor([[5.0]]).cuda()

result = model(test)

print("Input:", test.item())
print("Output predetto:", result.item())