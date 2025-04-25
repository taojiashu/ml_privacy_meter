import torch
from tqdm import tqdm
import wandb
import pdb
from tqdm import tqdm


def train_model(
    model,
    dataloader,
    max_epochs,
    criterion,
    optimizer,
    test_freq=None,
    testloader=None,
    device="cuda",
):
    model = model.to(device)
    model.train()
    for epoch in tqdm(range(max_epochs), desc="Epoch Progress", leave=True):
        correct, total = 0, 0
        running_loss = 0
        for data, label in tqdm(
            dataloader, desc=f"Training Epoch {epoch}", leave=False
        ):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total += label.shape[0]
                correct += (pred.argmax(-1) == label).sum().item()
                running_loss += loss.item()

        print(f"Epoch {epoch}: {running_loss:.4f}")
        print("Training accuracy is {:.3f}".format(correct / total))

        log_dict = {"training_loss": running_loss, "training_accuracy": correct / total}

        if test_freq is not None and testloader is not None:
            if epoch % test_freq == 0 or epoch == max_epochs - 1:
                test_accuracy = test_model(model, testloader, device)
                if wandb.run is not None:
                    log_dict["test_accuracy"] = test_accuracy

        if wandb.run is not None:
            wandb.log(log_dict)


def train_multilabel_model(
    model,
    dataloader,
    max_epochs,
    criterion,
    optimizer,
    test_freq=None,
    testloader=None,
    device="cuda",
):
    model = model.to(device)
    model.train()
    for epoch in tqdm(range(max_epochs), desc="Epoch Progress", leave=True):
        correct_per_attribute = torch.zeros(40, device=device)
        total_per_attribute = torch.zeros(40, device=device)
        running_loss = 0

        for data, label in tqdm(dataloader, desc="Batch Progress", leave=False):
            data = data.to(device)
            if isinstance(label, list):
                # Take the attribute labels from the list
                label = label[0].to(device).float()
            else:
                label = label.to(device)
            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                pred_binary = (
                    torch.sigmoid(pred) > 0.5
                ).float()  # Convert preds to binary
                correct = (pred_binary == label).float()  # Element-wise correctness
                correct_per_attribute += correct.sum(
                    0
                )  # Sum correct predictions for each attribute
                total_per_attribute += label.size(0)  # Count samples for averaging

        # Calculate accuracy per attribute and then average
        accuracy_per_attribute = correct_per_attribute / total_per_attribute
        average_accuracy = accuracy_per_attribute.mean().item()

        print(
            f"Epoch {epoch + 1}/{max_epochs}: Average Loss: {running_loss/len(dataloader)}, Average Accuracy: {average_accuracy}"
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "training_loss": running_loss,
                    "training_accuracy": average_accuracy,
                }
            )

        print(f"Epoch {epoch}: {running_loss:.4f}")
        print("Training accuracy is {:.3f}".format(average_accuracy))
        if test_freq is not None and testloader is not None:
            if epoch % test_freq == 0 or epoch == max_epochs - 1:
                test_accuracy = test_multilabel_model(model, testloader, device)
                if wandb.run is not None:
                    wandb.log({"epoch": epoch, "test_accuracy": test_accuracy})


def test_model(model, testloader, device):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    for data, label in testloader:
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred = model(data)
            total += label.shape[0]
            correct += (pred.argmax(-1) == label).sum().item()
    print("Test accuracy is {:.3f}".format(correct / total))

    return correct / total


def test_multilabel_model(model, testloader, device):
    model = model.to(device)
    model.eval()
    correct_per_attribute = torch.zeros(40, device=device)
    total_per_attribute = torch.zeros(40, device=device)

    for data, label in testloader:
        data = data.to(device)
        if isinstance(label, list):
            # Take the attribute labels from the list
            label = label[0].to(device).float()
        else:
            label = label.to(device)
        with torch.no_grad():
            pred = model(data)
            pred_binary = (torch.sigmoid(pred) > 0.5).float()  # Convert preds to binary
            correct = (pred_binary == label).float()  # Element-wise correctness
            correct_per_attribute += correct.sum(
                0
            )  # Sum correct predictions for each attribute
            total_per_attribute += label.size(0)  # Count samples for averaging

    # Calculate accuracy per attribute and then average
    accuracy_per_attribute = correct_per_attribute / total_per_attribute
    average_accuracy = accuracy_per_attribute.mean().item()

    print("Test accuracy is {:.3f}".format(average_accuracy))
    return average_accuracy
