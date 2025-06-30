import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel
from sklearn.metrics import accuracy_score

# PyTorch LightningModule: https://lightning.ai/docs/pytorch/latest/common/lightning_module.html

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-2):
        super().__init__()

        # stores all the provided args under the "self.hparams" attribute
        self.save_hyperparameters()
        # # equivalent
        # self.save_hyperparameters("model_name", "lr")
        # # accessible via 
        # self.hparams.model_name

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(vars(outputs).keys()) # to see only attributes (not methods) in outputs

        # last_hidden_state is (bs, seq_length, hidden_dim)
        h_cls = outputs.last_hidden_state[:,0] # (bs, hidden_dim)
        logits = self.W(h_cls) # (bs, 2)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"]) # (bs, num_classes)
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)
        val_acc = accuracy_score(batch["label"].cpu(), preds.cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
