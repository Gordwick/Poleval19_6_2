import torch
import torch.nn as nn
from transformers import  BertPreTrainedModel, BertModel, BertConfig

from app_config import CONFIG

class MyHerBertaModel(BertPreTrainedModel):
    def __init__(self, conf: BertConfig):
        super(MyHerBertaModel, self).__init__(conf)
        self.bert = BertModel.from_pretrained(CONFIG['PRETRAINED_MODEL'], config=conf)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.mx = nn.MaxPool1d(CONFIG['MAX_LEN'] - 1) # pools from n tokens to 1
        
        self.l0 = nn.Conv1d(768, 100, 2)
        self.drop_out1 = nn.Dropout(0.3)
        self.l1 = nn.Linear(100, 1536)
        self.drop_out2 = nn.Dropout(0.3)
        self.l2 = nn.Linear(1536, 768)
        self.drop_out3 = nn.Dropout(0.2)
        self.l3 = nn.Linear(768, 3)# output size
        
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
        torch.nn.init.normal_(self.l2.weight, std=0.02)
        torch.nn.init.normal_(self.l3.weight, std=0.02)
        
        self.activation0 = torch.nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.activation2 = torch.nn.LeakyReLU(negative_slope=0.05, inplace=False)
        self.activation3 = torch.nn.Softmax(dim=0)
    
    def forward(self, ids, mask, token_type_ids):
        out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        out = out[0]
        out = out.permute(0,2,1)

        out = self.activation0(self.l0(out))

        out = torch.squeeze(self.mx(out))

        out = self.drop_out1(out)
        out = self.activation1(self.l1(out))
        
        out = self.drop_out2(out)
        out = self.activation2(self.l2(out))

        out = self.drop_out3(out)
        out = self.activation3(self.l3(out))
        return out