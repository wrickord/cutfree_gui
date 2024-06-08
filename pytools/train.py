# Third-party imports
import torch


class Train:
    def __init__(
        self, model, optimizer, criterion, device,
        l1=False, l1_lambda=0.001, l2=False, l2_lambda=0.001
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.l1 = l1
        self.l1_lambda = l1_lambda
        self.l2 = l2
        self.l2_lambda = l2_lambda

    def get_class_weights(self, df):
        class_weights = df["Target"].value_counts(
            normalize=True
        ).sort_index().values
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    def train(self, loader):
        self.model.train() 
        epoch_loss = 0
        predictions, true_values = [], []
        for inputs, inputs_dims, targets in loader:            
            # Send data to GPU
            inputs = inputs.to(self.device)
            inputs_dims = inputs_dims.to(self.device)
            targets = targets.to(self.device)

            # Run model
            self.optimizer.zero_grad()
            outputs = self.model(inputs, inputs_dims)
            loss = self.criterion(outputs, targets)

            # L1 regularization
            if self.l1:
                l1_regularization = 0
                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                loss += self.l1_lambda * l1_regularization

            # L2 regularization
            if self.l2:
                l2_regularization = 0
                for param in self.model.parameters():
                    l2_regularization += torch.norm(param, p=2)
                loss += self.l2_lambda * l2_regularization

            # Backpropagation
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()

            # Get predictions and true values
            preds = outputs.argmax(dim=1, keepdim=True)
            predictions.extend(preds.cpu().numpy().flatten())
            true_values.extend(targets.cpu().numpy().flatten())

            # Delete data from GPU
            torch.detach(inputs)
            torch.detach(inputs_dims)
            torch.detach(targets)
            torch.detach(outputs)
            del inputs, inputs_dims, targets, outputs

        # Get correct predictions
        correct = sum(
            [pred == true for pred, true in zip(predictions, true_values)]
        )

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / len(predictions)
        return avg_loss, accuracy

    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0
        predictions, true_values = [], []
        with torch.no_grad(): # No need to track gradients during evaluation
            for inputs, inputs_dims, targets in loader:
                # Send data to GPU
                inputs = inputs.to(self.device)
                inputs_dims = inputs_dims.to(self.device)
                targets = targets.to(self.device)

                # Run model
                outputs = self.model(inputs, inputs_dims)
                loss = self.criterion(outputs, targets)
                epoch_loss += loss.item()

                # L1 regularization
                if self.l1:
                    l1_regularization = 0
                    for param in self.model.parameters():
                        l1_regularization += torch.norm(param, p=1)
                    loss += self.l1_lambda * l1_regularization

                # L2 regularization
                if self.l2:
                    l2_regularization = 0
                    for param in self.model.parameters():
                        l2_regularization += torch.norm(param, p=2)
                    loss += self.l2_lambda * l2_regularization

                # Get predictions and true values
                preds = outputs.argmax(dim=1, keepdim=True)
                predictions.extend(preds.cpu().numpy().flatten())
                true_values.extend(targets.cpu().numpy().flatten())

                # Delete data from GPU
                torch.detach(inputs)
                torch.detach(inputs_dims)
                torch.detach(targets)
                torch.detach(outputs)
                del inputs, inputs_dims, targets, outputs

        # Get correct predictions
        correct = sum(
            [pred == true for pred, true in zip(predictions, true_values)]
        )

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / len(predictions)
        return avg_loss, accuracy, true_values, predictions