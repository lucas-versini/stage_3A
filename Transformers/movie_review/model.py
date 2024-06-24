import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, embed_size, beta = 1., value = "identity"):
        """
        embed_size: int, size of the embedding
        beta: float, temperature parameter
        value: str, type of value matrix. If "identity", the value matrix is the identity matrix
        """
        super(SimpleAttention, self).__init__()
        self.beta = beta

        self.query = nn.Linear(embed_size, embed_size, bias = False)
        self.key = nn.Linear(embed_size, embed_size, bias = False)

        if value == "identity":
            self.value = nn.Linear(embed_size, embed_size, bias = False)
            self.value.weight = nn.Parameter(torch.eye(embed_size))
            self.value.requires_grad_(False)
            self.alpha = nn.Parameter(torch.randn(1))
        else:
            self.value = nn.Linear(embed_size, embed_size, bias = False)
            self.alpha = nn.Parameter(torch.tensor(1.))
            self.alpha.requires_grad_(False)
        self.value_type = value

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_size)
        queries = self.query(x)
        keys = self.key(x)
        values = self.alpha * self.value(x)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = torch.nn.functional.softmax(self.beta * scores, dim = -1)

        # Apply attention weights to values
        attended = torch.matmul(attention_weights, values)

        # Apply layer normalization with a residual connection
        y = attended + x
        x = y / y.norm(dim = -1, keepdim = True)
        return x

class CustomTransformer(nn.Module):
    def __init__(self, num_words, d, num_layers, beta = 1., share = True, value = "identity"):
        """
        num_words: int, number of words in the dictionary
        d: int, size of the embedding
        num_layers: int, number of layers
        beta: float, temperature parameter
        share: bool, if True, the attention layer is shared across layers
        value: str, type of value matrix. If "identity", the value matrix is the identity matrix
        """
        super(CustomTransformer, self).__init__()

        if share:
            self.shared_attention_layer = SimpleAttention(d, beta = beta, value = value)
            self.attention_layers = nn.ModuleList([self.shared_attention_layer for _ in range(num_layers)])
        else:
            self.attention_layers = nn.ModuleList([SimpleAttention(d, beta = beta, value = value) for _ in range(num_layers)])
        
        self.num_layers = num_layers
        self.encoder = nn.Parameter(torch.randn(num_words, d))
        self.proj = nn.Linear(d, 1)

    def forward(self, x, interm = False):
        x = self.encoder[x] # shape (batch_size, seq_len, d)
        x = x / x.norm(dim = -1, keepdim = True)

        if interm:
            interm_outputs = [x.clone()]

        for i in range(self.num_layers):
            x = self.attention_layers[i](x)
            if interm:
                interm_outputs.append(x.clone())

        x = torch.mean(x, dim = 1) # shape (batch_size, d)
        x = self.proj(x) # shape (batch_size, 1)
        x = torch.sigmoid(x).squeeze() # shape (batch_size)

        if interm:
            return x, interm_outputs
        return x