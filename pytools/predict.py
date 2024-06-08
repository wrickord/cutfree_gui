# Third-party imports
import torch

# Local application imports
from pytools.tokenizer import Tokenizer


class Predict:
    def __init__(self, 
                 model, 
                 device, 
                 max_seq_length, 
                 vocab, 
                 classes):
        self.model = model
        self.DEVICE = device
        self.MAX_SEQ_LENGTH = max_seq_length
        self.vocab = vocab
        self.classes = classes

    def predict(self, inputs, inputs_dims):
        # Get predictions
        with torch.no_grad():
            outputs = self.model(inputs, inputs_dims)
        preds = outputs.argmax(dim=1, keepdim=False)
            
        # Get targets and probabilities of predictions
        targets = [self.classes[pred] for pred in preds.cpu().numpy()]
        probs = [
            max(prob) for prob in torch.nn.functional.softmax(
                outputs.cpu(), dim=1
            ).numpy()
        ]

        # Delete data from GPU
        torch.detach(inputs)
        torch.detach(outputs)
        torch.detach(preds)
        del inputs, outputs, preds
        
        return targets, probs