# Standard library imports
import os
import sys
import math

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForTokenClassification

# Local application imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytools.config import DEVICE, SEED
from pytools.directory import SaveDirectory
from pytools.dataloaders import Dataloader
from pytools.tokenizer import Tokenizer
from pytools.transformer import Transformer
from pytools.mlp import MLP
from pytools.train import Train
from pytools.analyze import Analyze
from pytools.predict import Predict

# Constants
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 89
EARLY_STOPPING = 30
CUSTOM_MODEL = True
if CUSTOM_MODEL:
    HYPERPARAMETERS = {
        "input_dims": 64,
        "num_heads": 8,
        "num_layers": 1,
        "ff_dims": 64,
        "enc_dropout": 0.30,
        "num_mlp_layers": 2,
        "mlp_dims": [256, 128],
        "mlp_dropout": 0.20, 
        "optimizer_name": "AdamW",
        "lr": 1e-3,
        "epochs": 250
    }
else:
    MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" # "dslim/distilbert-NER"
    HYPERPARAMETERS = {
        "num_mlp_layers": 3,
        "mlp_dims": [512, 256, 128],
        "mlp_dropout": 0.20, 
        "optimizer_name": "AdamW",
        "lr": 1e-3,
        "epochs": 250
    }


class Model(nn.Module):
    def __init__(self, vocab=None, classes=None):
        super(Model, self).__init__()

        if CUSTOM_MODEL:
            self.transformer = Transformer(
                vocab_size=len(vocab), 
                input_dims=HYPERPARAMETERS["input_dims"], 
                num_heads=HYPERPARAMETERS["num_heads"],
                num_layers=HYPERPARAMETERS["num_layers"],
                ff_dims=HYPERPARAMETERS["ff_dims"],
                max_seq_length=MAX_SEQ_LENGTH,
                dropout=HYPERPARAMETERS["enc_dropout"]
            )
            mlp_input_dims = HYPERPARAMETERS["input_dims"] + 1
        else:
            self.transformer = AutoModelForTokenClassification.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            self.transformer = nn.Sequential(
                self.transformer.esm.embeddings,
                self.transformer.esm.encoder
            )
            mlp_input_dims = 512 + 1
        self.mlp = MLP(
            input_dims=mlp_input_dims,
            num_layers=HYPERPARAMETERS["num_mlp_layers"],
            mlp_dims=HYPERPARAMETERS["mlp_dims"],
            num_classes=len(classes),
            dropout=HYPERPARAMETERS["mlp_dropout"]
        )

    def forward(self, input, extra_input):
        if CUSTOM_MODEL:
            transformer_output = self.transformer(input)
            combined_output = torch.cat(
                (transformer_output, extra_input.unsqueeze(1)), 
                dim=1
            )
        else:
            transformer_output = torch.mean(
                self.transformer(input).last_hidden_state, 
                dim=1
            )
            combined_output = torch.cat(
                (transformer_output, extra_input.unsqueeze(1)), 
                dim=1
            )
        output = self.mlp(combined_output)
        
        return output
    

class CutFreeModel:
    def __init__(self, 
                 model_info=None,
                 load_model=False):
        self.CUR_DIR = CUR_DIR
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.EARLY_STOPPING = EARLY_STOPPING

        self.model_info = model_info
        self.load_model = load_model
        if load_model:
            self.load_dir = f"{self.CUR_DIR}/models/cutfree-v{model_info[0]}"
            self.vocab = torch.load(
                f"{self.load_dir}/vocab.pt"
            )
            self.classes = torch.load(
                f"{self.load_dir}/classes.pt"
            )
            self.hyperparameters = torch.load(
                f"{self.load_dir}/hyperparameters.pt"
            )
        else:
            self.save_dir = SaveDirectory("cutfree").get_dir()
            self.classes = np.array([0, 1])
            self.hyperparameters = HYPERPARAMETERS

    def get_data(self, predict=False, pred_df=None):
        if pred_df is None:
            pred_df = pd.read_csv(
                f"{self.CUR_DIR}/simulations/runtime_data.csv"
            )

        # Tokenize and load data
        inputs, inputs_dims, targets, self.vocab = Tokenizer(
            max_seq_length=self.MAX_SEQ_LENGTH
        ).get_data(pred_df, predict=predict)

        class_weights = pred_df["Target"].value_counts(normalize=True)
        self.class_weights = torch.tensor(
            [1.0 / class_weights[0],
                1.0 / class_weights[1]],
            dtype=torch.float
        ).to(DEVICE)

        if predict:
            return inputs, inputs_dims
        else:
            # Shuffle data
            inputs, inputs_dims, targets = shuffle(
                inputs, 
                inputs_dims,
                targets, 
                random_state=SEED
            )

            return inputs, inputs_dims, targets
    
    def save_info(self):
        torch.save(
            self.hyperparameters,
            f"{self.save_dir}/hyperparameters.pt"
        )
        torch.save(
            self.classes,
            f"{self.save_dir}/classes.pt"
        )
        torch.save(
            self.vocab,
            f"{self.save_dir}/vocab.pt"
        )

    def get_model(self, load_fold=False, predict=False):
        model = Model(self.vocab, self.classes).to(DEVICE)

        if self.load_model:
            model.load_state_dict(
                torch.load(
                    f"{self.load_dir}/model_{self.model_info[1]}.pt"
                )
            )
        elif load_fold:
            model.load_state_dict(
                torch.load(
                    f"{self.save_dir}/model_{load_fold}.pt"
                )
            )

        if predict:
            return model
        else:     
            optimizer_name = self.hyperparameters["optimizer_name"]
            lr = self.hyperparameters["lr"]
            optimizer = getattr(
                optim, 
                optimizer_name
            )(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss(
                weight= self.class_weights
            ).to(DEVICE)

            return model, optimizer, criterion
    
    def train(self):
        inputs, inputs_dims, targets = self.get_data()
        self.hyperparameters["vocab_size"] = len(self.vocab)

        # Save hyperparameters, classes, and vocab using torch
        self.save_info()

        # Train model
        N_SPLITS = 10
        EPOCHS = self.hyperparameters["epochs"]
        train_losses = [[] for _ in range(N_SPLITS)]
        val_losses = [[] for _ in range(N_SPLITS)]
        test_accs = []
        for fold, (train_val_idx, test_idx) in enumerate(
            KFold(n_splits=N_SPLITS).split(inputs)
        ):
            # Print progress
            fold += 1
            print(f"\nFold {fold}/{N_SPLITS}")
            print(f"-" * 10)

            # Get training, validation, and testing loaders
            D = Dataloader(DEVICE, self.BATCH_SIZE, SEED)
            train_loader, val_loader = D.get_train_loaders(
                inputs[train_val_idx], 
                inputs_dims[train_val_idx],
                targets[train_val_idx]
            )
            test_loader = D.get_test_loader(
                inputs[test_idx], 
                inputs_dims[test_idx],
                targets[test_idx]
            )

            # Get model, optimizer, and criterion
            model, optimizer, criterion = self.get_model()

            # Train model
            T = Train(model, optimizer, criterion, DEVICE)
            best_loss = math.inf
            selected_epoch = 0
            for epoch in range(EPOCHS):                    
                train_loss, train_acc = T.train(train_loader)
                val_loss, val_acc = T.evaluate(val_loader)[:2]
                if epoch == 0:
                    best_loss = val_loss
                    selected_epoch = epoch
                    torch.save(
                        model.state_dict(), 
                        f"{self.save_dir}/model_{fold}.pt"
                    )
                elif val_loss <= best_loss and val_loss <= (train_loss * 1.01):
                    best_loss = val_loss
                    selected_epoch = epoch
                    torch.save(
                        model.state_dict(), 
                        f"{self.save_dir}/model_{fold}.pt"
                    )
                elif epoch - selected_epoch > self.EARLY_STOPPING:
                    break
                
                print(f"Epoch {epoch + 1}")
                print(f"\tTrain Loss: {train_loss:.3f}")
                print(f"\tTrain Accuracy: {train_acc * 100:.3f}")
                print(f"\tValidation Loss: {val_loss:.3f}")
                print(f"\tValidation Accuracy: {val_acc * 100:.3f}")

                # Save for plotting
                train_losses[fold-1].append(train_loss)
                val_losses[fold-1].append(val_loss)

            # Load current fold of model
            model, optimizer, criterion = self.get_model(
                load_fold=fold
            )

            # Get predictions from current fold
            P = Train(model, optimizer, criterion, DEVICE)
            _, test_acc, true_vals, preds = P.evaluate(test_loader)
            test_accs.append(test_acc)
            print(f"\nTest Accuracy: {test_acc * 100:.3f}\n")

            # Analyze model results
            A = Analyze(
                fig_size=(15, 10),
                fold=fold, 
                save_dir=self.save_dir, 
            )
            A.get_loss_curves(train_losses, val_losses, selected_epoch)
            A.get_cm(self.classes, true_vals, preds, accuracy=test_acc * 100)
            
        print(f"Average accuracy: {np.mean(test_accs) * 100:.3f}")

    def predict(self, inputs, inputs_dims):
        model = self.get_model(predict=True)
        model.to(DEVICE)
        model.eval()

        P = Predict(
            model=model, 
            device=DEVICE, 
            max_seq_length=self.MAX_SEQ_LENGTH, 
            vocab=self.vocab, 
            classes=self.classes
        )       
        sites_df = pd.DataFrame(
            data={
                "Input": inputs, 
                "Oligo_Input": inputs_dims
            },
            index=range(len(inputs))
        )
        sites_df["Oligo_Input"] = sites_df["Oligo_Input"].apply(lambda x: len(x))
        sites, oligo_dims = self.get_data(predict=True, pred_df=sites_df)
        pred_dataset = torch.utils.data.TensorDataset(
            sites.clone().detach(), 
            oligo_dims.clone().detach()
        )
        pred_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False
        )

        preds, probs = [], []
        for inputs, inputs_dims in pred_loader:
            inputs = inputs.to(DEVICE)
            inputs_dims = inputs_dims.to(DEVICE)
            pred = P.predict(inputs, inputs_dims)
            preds.extend(pred[0])
            probs.extend(pred[1])

        return preds, probs