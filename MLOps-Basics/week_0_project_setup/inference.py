import torch
from model import ColaModel
from data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]
        self.device = next(self.model.parameters()).device
    
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        input_ids = torch.tensor([processed["input_ids"]], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([processed["attention_mask"]], dtype=torch.long).to(self.device)

        logits = self.model(input_ids, attention_mask) # (bs, 2)
        scores = self.softmax(logits[0]) # (2) or (num_classes)

        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions

if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/epoch=3-step=1072.ckpt")
    print(predictor.predict(sentence))  
