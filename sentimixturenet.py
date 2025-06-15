class SentimixtureNet:
    def __init__(self):
        import torch.nn as nn
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

        class Wrapper(nn.Module):
            def __init__(self, encoder, classifier):
                super().__init__()
                self.encoder = encoder
                self.classifier = classifier

            def forward(self, input_ids, attention_mask):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                return self.classifier(pooled_output)

        self.model = Wrapper(self.encoder, self.classifier)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self.model

