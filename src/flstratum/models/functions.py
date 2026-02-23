import torch
import torch.nn.functional as F
import copy


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def train(model, dataloader, epochs, lr=3e-4, device=DEVICE):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = model(x)

            # Calculate loss
            loss = criterion(output, y)
            train_loss += loss.item()
            train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

            # Zero grad
            optimizer.zero_grad()
            
            # Backprog
            loss.backward()

            # Optimize
            optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_fedprox(model, dataloader, epochs, mu=0.01, lr=3e-4, device=DEVICE):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_model = copy.deepcopy(model)

    model.train()

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = model(x)

            # Calculate loss
            loss = criterion(output, y) + proximal_term(model, global_model, mu=mu)
            train_loss += loss.item()
            train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

            # Zero grad
            optimizer.zero_grad()

            # Backprog
            loss.backward()

            # Optimize
            optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy


def proximal_term(model, global_model, mu):
    """Compute the FedProx proximal term."""
    proximal_term = 0.0
    for param, global_param in zip(model.parameters(), global_model.parameters()):
        proximal_term += (param - global_param).norm(2)
    return (mu / 2) * proximal_term


def train_ditto(personalized_model, global_model, dataloader, epochs, lr=3e-4, lambda_=0.5, device=DEVICE):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(personalized_model.parameters(), lr=lr)

    personalized_model.train()

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = personalized_model(x)

            # Calculate loss
            prox = 0
            for param, global_param in zip(personalized_model.parameters(), global_model.parameters()):
                prox += (param - global_param).norm(2)
            loss = criterion(output, y) + lambda_ * prox
            train_loss += loss.item()
            train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

            # Zero grad
            optimizer.zero_grad()

            # Backprog
            loss.backward()

            # Optimize
            optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy


def train_fedmix(models, weights, dataloader, epochs, lr=3e-4, device=DEVICE):
    criterion = torch.nn.CrossEntropyLoss()
    params = [weights]
    for model in models:
        params += list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    k = len(models)

    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = torch.stack([m(x) for m in models])
            pred = (weights.softmax(0).view(k, 1, 1) * output)

            # Calculate loss
            loss = criterion(pred, y)
            train_loss += loss.item()
            train_accuracy += accuracy_fn(y, torch.argmax(output, dim=1))

            # Zero grad
            optimizer.zero_grad()
            
            # Backprog
            loss.backward()

            # Optimize
            optimizer.step()

    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy



def evaluate(model, dataloader, device=DEVICE):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = model(x)

            # Compute score
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def evaluate_fedmix(models, weights, dataloader, device=DEVICE):
    for model in models:
        model.eval()
    for w in weights:
        w.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x = x.to(device)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)  # Add channel dimension for MNIST
            y = y.to(device)

            # Forward pass
            output = torch.stack([m(x) for x in models])
            pred = (w.softmax(0).view(k, 1, 1) * output)

            # Compute score
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total