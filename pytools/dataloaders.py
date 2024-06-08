# Third-party imports
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch


class Dataloader:
    def __init__(self, DEVICE, BATCH_SIZE=16, RANDOM_STATE=None):
        self.BATCH_SIZE = BATCH_SIZE
        self.RANDOM_STATE = RANDOM_STATE
        self.DEVICE = DEVICE
        if RANDOM_STATE == None:
            self.RANDOM_STATE = np.random.randint(0, 100)            

    def get_train_loaders(self, inputs, inputs_dims, targets):
        # Train test split
        inputs_train, inputs_val, inputs_dims_train, inputs_dims_val, \
            targets_train, targets_val = train_test_split(
                inputs, 
                inputs_dims, 
                targets, 
                test_size=0.2, 
                random_state=self.RANDOM_STATE
            )

        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(
            inputs_train.clone().detach(), 
            inputs_dims_train.clone().detach(),
            targets_train.clone().detach()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        val_dataset = torch.utils.data.TensorDataset(
            inputs_val.clone().detach(), 
            inputs_dims_val.clone().detach(),
            targets_val.clone().detach()
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        return train_loader, val_loader
    
    def get_test_loader(self, inputs, inputs_dims, targets):
        test_dataset = torch.utils.data.TensorDataset(
            inputs.clone().detach(), 
            inputs_dims.clone().detach(),
            targets.clone().detach()
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        return test_loader
    
    def get_class_weights(self, loader, classes):
        temp_targets = []
        for _, targets in loader:
            temp_targets.extend(targets.cpu().numpy())
        targets = [classes[temp_target] for temp_target in temp_targets]
        class_weights = torch.Tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(targets),
                y=targets
            )
        ).to(self.DEVICE)
        return class_weights