import torch
import numpy as np
from preprocess import Preprocess
from app_config import CONFIG


class Predictor:
    def __init__(self, tokenizer, model):
        self.Preprocess = Preprocess(tokenizer)
        self.model = model

    def predict(self, text: str) -> np.ndarray:
        """
        Run model and get result
        :return: numpy array of model output
        """

        X = self.Preprocess.preprocess(text)

        with torch.no_grad():
            ids = X['ids'].to(CONFIG['DEVICE'], dtype=torch.long)
            token_type_ids = X['token_type_ids'].to(CONFIG['DEVICE'], dtype=torch.long)
            mask = X['mask'].to(CONFIG['DEVICE'], dtype=torch.long)

            y_pred = self.model(
                ids=torch.stack([ids]),
                mask=torch.stack([mask]),
                token_type_ids=torch.stack([token_type_ids])
                )

        # convert result to a numpy array on CPU
        y_pred = y_pred.cpu().numpy()

        return y_pred