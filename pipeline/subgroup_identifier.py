from models import *
import torch
from torch import optim


class Grouper:
    def __init__(self):
        self.models = []

    def eval_grouper(self, dataloaders, model, loss_threshold, expected_label):
        """
        Using the validation set, if the loss exceeds the loss threshold
        and the predicted label for the pt is wrong and the ground truth label
        is what we want rn, put in group 1, else 0.

        i.e. for the data pts that the model currently gets SUPER wrong, if the pt
        is actually the label we're working with, separate it out into group 1.
        put everything else in group 0.

        Then, see how well the grouper model performs at capturing this separation.
        """

        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in dataloaders["val"]:
                embeddings = batch["embeddings"]
                losses = batch["loss"]
                predicted_labels = batch["predicted_labels"]
                actual_labels = batch["actual_labels"]

                group_ids = [
                    1
                    if loss >= loss_threshold
                    and predicted != actual
                    and actual == expected_label
                    else 0
                    for loss, predicted, actual in zip(
                        losses, predicted_labels, actual_labels
                    )
                ]

                embeddings = embeddings.view(embeddings.size(0), -1)
                outputs = model(embeddings)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == group_ids).sum().item()
                total += group_ids.size(0)

        accuracy = correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def LR_grouper(
        self,
        device,
        dataloaders,
        img_sz,
        num_classes,
        epochs,
        lr,
        loss_threshold,
        expected_label,
    ):
        model = LogisticRegression(img_sz, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.SGD(model.parameters(), lr=lr)

        for _ in range(epochs):
            for batch in dataloaders["train"]:
                embeddings = batch["embeddings"]
                losses = batch["loss"]
                predicted_labels = batch["predicted_label"]
                actual_labels = batch["actual_label"]

                N = embeddings.shape[0]

                group_ids = [
                    1
                    if loss >= loss_threshold
                    and predicted != actual
                    and actual == expected_label
                    else 0
                    for loss, predicted, actual in zip(
                        losses, predicted_labels, actual_labels
                    )
                ]

                group_ids = torch.tensor(group_ids).to(torch.long).to(device)

                embeddings = embeddings.view(N, -1)
                outputs = model(embeddings)
                loss = torch.mean(criterion(outputs, group_ids))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        accuracy = self.eval_grouper(dataloaders, model, loss_threshold, expected_label)
        return model, accuracy

