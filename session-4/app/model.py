import torch.nn as nn


class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # EmbeddingBag is the combination of Embedding and mean() in a single layer
        # TODO complete the embedding bad and fc layers with the correct parameters. Set `sparse`=True in the EmbeddingBag
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.dout = nn.Dropout(0.4)
        self.fc = nn.Linear( 
            in_features=embed_dim, 
            out_features=num_class
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # bsz, slen, vocab = text.shape
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
