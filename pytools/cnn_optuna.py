# Standard library imports
import os
import math

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState

# Local application imports
from pytools.dataloaders import Dataloader
from pytools.cnn import CNN
from pytools.mlp import MLP
from pytools.train import Train


class Optuna:
    def __init__(self, 
                 inputs, 
                 targets, 
                 classes, 
                 save_dir, 
                 DEVICE,
                 EARLY_STOPPING,
                 BATCH_SIZE,
                 RANDOM_STATE,
                 VOCAB_SIZE,
                 MAX_SEQ_LENGTH,
                 NUM_CLASSES):
        super(Optuna, self).__init__()

        self.inputs = inputs
        self.targets = targets
        self.classes = classes
        self.save_dir = save_dir
        self.DEVICE = DEVICE
        self.EARLY_STOPPING = EARLY_STOPPING
        self.BATCH_SIZE = BATCH_SIZE
        self.RANDOM_STATE = RANDOM_STATE
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.NUM_CLASSES = NUM_CLASSES

    def define_model(self, trial):
        encoder = CNN(
            device = self.DEVICE,
            batch_size=self.BATCH_SIZE,
            vocab_size=self.VOCAB_SIZE,
            embedding_dims=trial.suggest_categorical(
                "embedding_dims", [8, 16, 32, 64, 128]
            ),
            max_seq_length=self.MAX_SEQ_LENGTH,
            num_layers=1,
            cnn_layer_dims=trial.suggest_categorical(
                "cnn_layer_dims", [16, 32, 64, 128, 256, 512]
            ),
            kernel_size=trial.suggest_categorical(
                "kernel_size", [5, 7]
            ),
            pooling_size=2,
            stride=trial.suggest_int("stride", 1, 3),
            dropout=trial.suggest_float("cnn_dropout", 0.1, 0.5)
        )
        output_dims = encoder.get_output_dims()
        decoder = MLP(
            input_dims=output_dims,
            num_layers=2,
            mlp_dims=[
                trial.suggest_categorical(
                    "mlp_dims1", [16, 32, 64, 128, 256, 512, 1024, 2048]
                ),
                trial.suggest_categorical(
                    "mlp_dims2", [16, 32, 64, 128, 256, 512, 1024, 2048]
                )
            ],
            num_classes=self.NUM_CLASSES,
            dropout=trial.suggest_float("mlp_dropout", 0.1, 0.5)
        )
        model = nn.Sequential(
            encoder,
            decoder
        ).to(self.DEVICE)

        return model

    def objective(self, trial):
        # Get training and validation loaders
        D = Dataloader(self.DEVICE, self.BATCH_SIZE, self.RANDOM_STATE)
        train_loader, val_loader = D.get_train_loaders(
            self.inputs, 
            self.targets
        )

        # Generate the model
        model = self.define_model(trial).to(self.DEVICE)

        # Generate the optimizer and loss function
        optimizer_name = trial.suggest_categorical(
            "optimizer", 
            ["RMSprop", "AdamW", "Adam"]
        )
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(self.DEVICE) 

        # Set the epochs
        self.EPOCHS = 100

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
                trial.report(val_loss, epoch)
                break

            # Handle pruning based on the intermediate value
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        # Delete data from GPU
        model.to(torch.device("cpu"))
        del model, optimizer
        torch.cuda.empty_cache()

        return val_loss

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
            study_name="rbs_cnn",
        )
        study.optimize(self.objective, n_trials=50)

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
            "cnn_layer_dims": trial.params["cnn_layer_dims"],
            "stride": trial.params["stride"],
            "cnn_dropout": trial.params["cnn_dropout"],
            "mlp_dims": [
                trial.params["mlp_dims1"],
                trial.params["mlp_dims2"]
            ],
            "mlp_dropout": trial.params["mlp_dropout"],
            "optimizer": trial.params["optimizer"],
            "lr": trial.params["lr"],
            "epochs": self.EPOCHS
        }