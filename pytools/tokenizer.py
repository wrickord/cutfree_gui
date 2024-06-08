# Third-party imports
import numpy as np
import pandas as pd
import torch
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:
    def __init__(self, max_seq_length):
        self.MAX_SEQ_LENGTH = max_seq_length

    def build_vocab(self, data):
        return build_vocab_from_iterator(
            map(list, data)
        )

    def process_data(self, raw_data, vocab):
        # Tokenize data
        data = [[torch.tensor(
            [vocab[token] for token in list(
                item[max(-1*len(item), -1*self.MAX_SEQ_LENGTH):]
            )],
            dtype=torch.long
        )] for item in raw_data]

        # Pad data to help with batching
        padded_data = [torch.cat((
            item[0],
            torch.ones(
                # (1, int(self.MAX_SEQ_LENGTH - len(item[0]))), 
                int(self.MAX_SEQ_LENGTH - len(item[0])),
                dtype=torch.long
            )
        )) for item in data]

        return torch.stack(padded_data)

    def get_data(self, df, predict=False):
        if predict:
            # Build vocab and process inputs
            inputs = df["Input"].astype("str").values
            inputs_dims = torch.tensor(df["Oligo_Input"].values)
            vocab = self.build_vocab(inputs)
            processed_inputs = self.process_data(inputs, vocab)

            return processed_inputs, inputs_dims, None, vocab
        
        # Sort data by name
        df.sort_values(by=["Target"], inplace=True)

        # Build vocab and process inputs
        inputs = df["Input"].astype("str").values
        inputs_dims = torch.tensor(df["Oligo_Input"].values)
        vocab = self.build_vocab(inputs)
        processed_inputs = self.process_data(inputs, vocab)

        # Transform target sequences into tensors
        targets = torch.tensor(
            pd.factorize(df["Target"].values)[0], 
            dtype=torch.long
        )

        return processed_inputs, inputs_dims, targets, vocab