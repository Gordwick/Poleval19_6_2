from app_config import CONFIG
from transformers import AutoTokenizer

class Preprocess:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def preprocess(self, text: str) -> dict:
        """Interface for text sample preprocessing"""
        text = text.replace("@anonymized_account", "@osoba")
        
        text = ' '.join(text.split())
        enc = self.tokenizer(text, 
                             max_length=CONFIG['MAX_LEN'], 
                             truncation=True, 
                             padding='max_length', 
                             return_tensors='pt'
                             )

        return {
            'ids': enc.input_ids[0],
            'mask': enc.attention_mask[0],
            'token_type_ids': enc.token_type_ids[0],
        } 