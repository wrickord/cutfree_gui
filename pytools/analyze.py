# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Analyze:
    def __init__(self, fig_size=(20, 20), save_dir=-1, fold=-1):
        self.fig_size = fig_size
        self.save_dir = save_dir
        self.fold = fold

    def get_target_distribution(self, df):
        target_dist = df["Target"].value_counts()
        target_dist = target_dist.sort_index()
        target_dist.plot.bar(figsize=self.fig_size)
        plt.title("Target Distribution")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.tight_layout()
        if self.save_dir != -1:
            plt.savefig(
                f"{self.save_dir}/target_distribution.png",
                bbox_inches="tight"
            )
        else:
            plt.savefig("target_distribution.png", bbox_inches="tight")

    def get_loss_curves(self, train_losses, val_losses, selected_epoch):
        plt.figure(figsize=self.fig_size)
        if self.fold == -1:
            plt.plot(train_losses, label="Training loss")
            plt.plot(val_losses, label="Validation loss")
        else:
            plt.plot(train_losses[self.fold-1], label="Training loss")
            plt.plot(val_losses[self.fold-1], label="Validation loss")
        plt.axvline(
            x=selected_epoch, 
            color="r", 
            linestyle="--", 
            label="Selected epoch"
        )
        plt.legend()
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        if self.save_dir != -1:
            if self.fold == -1:
                plt.savefig(
                    f"{self.save_dir}/loss_curves.png", 
                    bbox_inches="tight"
                )
            else:
                plt.savefig(
                    f"{self.save_dir}/loss_curves_{self.fold}.png",
                    bbox_inches="tight"
                )
        else:
            plt.savefig("loss_curves.png", bbox_inches="tight")

    def get_cm(self, classes, true_values, predictions, accuracy=-1):
        # Initialize dataframe with all possible targets
        cm_df = pd.DataFrame(
            np.zeros((len(classes), len(classes))), 
            index=classes, 
            columns=classes
        )

        # Add values to dataframe
        for i in range(len(classes)):
            cm_df.iloc[i] = confusion_matrix(
                classes[true_values], 
                classes[predictions], 
                labels=classes
            )[i]

        # Normalize dataframe
        col_sum = np.sum(cm_df, axis=0)
        col_sum[col_sum == 0] = 1
        norm_cm = cm_df / col_sum

        # Plot confusion matrix
        plt.figure(figsize=self.fig_size)
        ax = sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues")
        ax.set(
            xlabel="Predicted", 
            ylabel="Actual", 
            title="Confusion Matrix" if accuracy == -1 \
                else f"Confusion Matrix (Accuracy: {accuracy:.3f}%)"
        )
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        if self.save_dir != -1:
            if self.fold == -1:
                plt.savefig(
                    f"{self.save_dir}/confusion_matrix.png",
                    bbox_inches="tight"
                )
            else:
                plt.savefig(
                    f"{self.save_dir}/confusion_matrix_{self.fold}.png",
                    bbox_inches="tight"
                )
        else:
            plt.savefig("confusion_matrix.png", bbox_inches="tight")