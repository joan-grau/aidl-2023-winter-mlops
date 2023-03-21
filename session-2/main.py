import torch

from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, dataloader, criterion, optimizer):
    model.train()
    loss_total = 0
    acc = 0

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        acc += accuracy(labels, outputs)

    return loss_total / len(dataloader), acc / len(dataloader)

def eval_single_epoch(model, dataloader, criterion):
    model.eval()
    loss_total = 0
    acc = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item()
            acc += accuracy(labels, outputs)

    return loss_total / len(dataloader), acc / len(dataloader)


def train_model(config):
    
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    my_dataset = MyDataset("./mnist/samples/", "./mnist/chinese_mnist.csv", transforms.Compose(transform))

    train_size, val_size, test_size = 10000, 2500, 2500
    train_set, val_set, test_set = random_split(my_dataset, [train_size, val_size, test_size]) # Improve getting balanced samples for each character
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])
    
    my_model = MyModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config["lr"])

    best_val_acc = 0
    print(f"Star training using: {device.type}")

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(my_model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_single_epoch(my_model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_scripted = torch.jit.script(my_model) # Export to TorchScript
            model_scripted.save("./models/best_model.pt") # Save

    best_model = torch.jit.load("./models/best_model.pt")

    test_loss, test_acc = eval_single_epoch(best_model, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    return best_model


if __name__ == "__main__":

    config = {
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 10,
    }
    train_model(config)
