# Standard library imports
import os
import math

# Third-party imports
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState

# Local application imports
from pytools.config import DEVICE, SEED
from pytools.dataloaders import Dataloader
from pytools.transformer import Transformer
from pytools.mlp import MLP
from pytools.train import Train


class Model(nn.Module):
    def __init__(self, 
                 hyperparameters, 
                 max_seq_length, 
                 vocab_size, 
                 classes=None):
        super(Model, self).__init__()

        self.transformer = Transformer(
            vocab_size=vocab_size, 
            input_dims=hyperparameters["input_dims"], 
            num_heads=hyperparameters["num_heads"],
            num_layers=hyperparameters["num_layers"],
            ff_dims=hyperparameters["ff_dims"],
            max_seq_length=max_seq_length,
            dropout=hyperparameters["enc_dropout"]
        )
        mlp_input_dims = hyperparameters["input_dims"] + 1
        
        self.mlp = MLP(
            input_dims=mlp_input_dims,
            num_layers=hyperparameters["num_mlp_layers"],
            mlp_dims=hyperparameters["mlp_dims"],
            num_classes=len(classes),
            dropout=hyperparameters["mlp_dropout"]
        )

    def forward(self, input, extra_input):
        transformer_output = self.transformer(input)
        combined_output = torch.cat(
            (transformer_output, extra_input.unsqueeze(1)), 
            dim=1
        )
        output = self.mlp(combined_output)
        
        return output
    

class TransformerOptuna:
    def __init__(
        self, inputs, inputs_dims, targets, classes, weights, save_dir, DEVICE, 
        EARLY_STOPPING, BATCH_SIZE, RANDOM_STATE, VOCAB_SIZE, MAX_SEQ_LENGTH, 
        NUM_CLASSES
    ):
        self.inputs = inputs
        self.inputs_dims = inputs_dims
        self.targets = targets
        self.classes = classes
        self.class_weights = weights
        self.save_dir = save_dir
        self.DEVICE = DEVICE
        self.EARLY_STOPPING = EARLY_STOPPING
        self.BATCH_SIZE = BATCH_SIZE
        self.RANDOM_STATE = RANDOM_STATE
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.NUM_CLASSES = NUM_CLASSES

    def define_model(self, trial):
        # Get hyperparameters
        input_dims = trial.suggest_categorical(
            "input_dims",
            [32, 64, 128, 256, 512]
        )
        num_heads=trial.suggest_categorical(
            "num_heads", 
            [4, 8, 16, 32]
        )
        num_layers=trial.suggest_categorical(
            "num_layers", 
            [1, 2, 3, 4]
        )
        ff_dims=trial.suggest_categorical(
            "ff_dims", 
            [8, 16, 32, 64, 128, 256]
        )
        dropout=trial.suggest_float("enc_dropout", 0, 0.8, step=0.05)
        num_layers=trial.suggest_categorical(
            "num_mlp_layers", 
            [1, 2, 3, 4, 5, 6]
        )
        mlp_dims=[
            trial.suggest_categorical(
                "mlp_dims1", 
                [16, 32, 64, 128, 256, 512, 1024, 2048]
            ),
            trial.suggest_categorical(
                "mlp_dims2", 
                [16, 32, 64, 128, 256, 512, 1024, 2048]
            ),
            trial.suggest_categorical(
                "mlp_dims3", 
                [16, 32, 64, 128, 256, 512, 1024, 2048]
            ),
            trial.suggest_categorical(
                "mlp_dims4", 
                [16, 32, 64, 128, 256, 512, 1024, 2048]
            ),
            trial.suggest_categorical(
                "mlp_dims5", 
                [16, 32, 64, 128, 256, 512, 1024, 2048]
            ),
            trial.suggest_categorical(
                "mlp_dims6", 
                [16, 32, 64, 128, 256, 512, 1024, 2048]
            )
        ]
        dropout=trial.suggest_float("mlp_dropout", 0, 0.8, step=0.05)

        hyperparameters = {
            "input_dims": input_dims,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dims": ff_dims,
            "enc_dropout": dropout,
            "num_mlp_layers": num_layers,
            "mlp_dims": mlp_dims,
            "mlp_dropout": dropout
        }

        model = Model(
            hyperparameters, 
            self.MAX_SEQ_LENGTH, 
            self.VOCAB_SIZE, 
            self.classes
        ).to(DEVICE)

        return model

    def objective(self, trial):
        # Train model
        N_SPLITS = 5
        test_losses = []
        for fold, (train_val_idx, test_idx) in enumerate(
            KFold(n_splits=N_SPLITS).split(self.inputs)
        ):
            # Print progress
            fold += 1
            print(f"Fold {fold}/{N_SPLITS}")
            print(f"-" * 10)

            # Get training, validation, and testing loaders
            D = Dataloader(self.DEVICE, self.BATCH_SIZE, self.RANDOM_STATE)
            train_loader, val_loader = D.get_train_loaders(
                self.inputs[train_val_idx], 
                self.inputs_dims[train_val_idx],
                self.targets[train_val_idx]
            )
            test_loader = D.get_test_loader(
                self.inputs[test_idx], 
                self.inputs_dims[test_idx],
                self.targets[test_idx]
            )

            # Generate the model
            model = self.define_model(trial).to(self.DEVICE)

            # Generate the optimizer and loss function
            optimizer_name = trial.suggest_categorical(
                "optimizer", 
                ["RMSprop", "AdamW", "Adam", "SGD"]
            )
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            optimizer = getattr(
                optim, 
                optimizer_name
            )(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights
            ).to(self.DEVICE) 

            # Set the epochs
            self.EPOCHS = 250

            # Train model
            T = Train(model, optimizer, criterion, self.DEVICE)
            best_loss = math.inf
            selected_epoch = 0
            for epoch in range(self.EPOCHS):
                train_loss = T.train(train_loader)[0]
                val_loss = T.evaluate(val_loader)[0]
                if val_loss <= best_loss and val_loss <= (train_loss * 1.01):
                    best_loss = val_loss
                    selected_epoch = epoch
                elif epoch - selected_epoch > self.EARLY_STOPPING:
                    break

                # Handle pruning based on the intermediate value
                trial.report(val_loss, epoch + (fold * self.EPOCHS))
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # Calculate test loss and accuracy
            test_loss, test_acc = T.evaluate(test_loader)[:2]
            test_losses.append(test_loss)
            print(f"Test loss: {test_loss}")
            print(f"Test accuracy: {test_acc}")
    
        # Delete data from GPU
        model.to(torch.device("cpu"))
        del model, optimizer
        torch.cuda.empty_cache()

        # Get average test loss
        test_loss = sum(test_losses) / N_SPLITS
        
        return test_loss

    def analyze_study(self, study, trial):
        os.mkdir(f"{self.save_dir}/optuna")

        # Save best hyperparameters
        with open(f"{self.save_dir}/optuna/best_params.txt", "w") as f:
            f.write(f"Best trial:\n")
            f.write(f"\tValue: {trial.value}\n")
            for key, value in trial.params.items():
                f.write(f"\t{key}: {value}\n")

        # Save study
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{self.save_dir}/optuna/plot_optimization_history.html")

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{self.save_dir}/optuna/plot_parallel_coordinate.html")

        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f"{self.save_dir}/optuna/plot_slice.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{self.save_dir}/optuna/plot_param_importances.html")

        fig = optuna.visualization.plot_edf(study)
        fig.write_html(f"{self.save_dir}/optuna/plot_edf.html")

        fig = optuna.visualization.plot_contour(
            study, 
            params=["num_heads", "num_layers"]
        )
        fig.write_html(f"{self.save_dir}/optuna/plot_contour.html")

        fig = optuna.visualization.plot_intermediate_values(study)
        fig.write_html(f"{self.save_dir}/optuna/plot_intermediate_values.html")

    def run_optuna(self):
        study = optuna.create_study(
            direction="minimize", 
            study_name="cutfree_transformer"
        )
        study.optimize(self.objective, n_trials=150)

        pruned_trials = study.get_trials(
            deepcopy=False, 
            states=[TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, 
            states=[TrialState.COMPLETE]
        )

        print("Study statistics: ")
        print("\tNumber of finished trials: ", len(study.trials))
        print("\tNumber of pruned trials: ", len(pruned_trials))
        print("\tNumber of complete trials: ", len(complete_trials))        
        
        # Print best trial
        trial = study.best_trial
        print("Best trial:")
        print("\tValue: ", trial.value)
        for key, value in trial.params.items():
            print("\t{}: {}".format(key, value))
        self.analyze_study(study, trial)
        return {
            "input_dims": trial.params["input_dims"],
            "num_heads": trial.params["num_heads"],
            "num_layers": trial.params["num_layers"],
            "ff_dims": trial.params["ff_dims"],
            "enc_dropout": trial.params["enc_dropout"],
            "num_mlp_layers": trial.params["num_mlp_layers"],
            "mlp_dims": [
                trial.params["mlp_dims1"],
                trial.params["mlp_dims2"],
                trial.params["mlp_dims3"],
                trial.params["mlp_dims4"],
                trial.params["mlp_dims5"],
                trial.params["mlp_dims6"]
            ],
            "mlp_dropout": trial.params["mlp_dropout"],
            "optimizer_name": trial.params["optimizer"],
            "lr": trial.params["lr"],
            "epochs": self.EPOCHS
        }