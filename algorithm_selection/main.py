# System imports
import os
import sys

# Third-party imports
import torch

# Local application imports
from cutfree_model import CutFreeModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pytools.config import DEVICE, SEED

# Clear GPU cache
torch.cuda.empty_cache()

# Constants
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {DEVICE}\nSeed: {SEED}")
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # For training
    train = True
    if train:
        optuna = False
        model = CutFreeModel()
        model.train(optuna=optuna)

    predict = not train
    if predict:
        model_info = [1, 1] # Version, model number 
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "predictions", 
            f"v{model_info[0]}_{model_info[1]}"
        )

        starting_oligo = ["NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"]
        restriction_sites = ["GGGAC GTCTC GGNCC GTNAC CCNGG"]
        
        # Predict if rbs is present
        model = CutFreeModel(
            load_model=True, 
            model_info=model_info 
        )
        preds, probs = model.predict(restriction_sites, starting_oligo)

        print(preds, probs)

"""
nohup python -u main.py > output.log 2>&1 &
ps -uwrickord | grep python
"""