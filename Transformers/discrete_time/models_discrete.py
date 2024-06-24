import torch
import torch.nn as nn

class SimpleAttentionIdentity(nn.Module):
    def __init__(self, embed_size, beta = 1., value = "general", alpha = None):
        """
        value: 'id' (V = alpha I_d) or 'general'.
        If value == 'id', alpha gives the default value. If alpha is None, alpha will be randomly initialized.
        """
        super(SimpleAttentionIdentity, self).__init__()
        self.beta = beta
        self.query = nn.Linear(embed_size, embed_size, bias = False)
        self.key = nn.Linear(embed_size, embed_size, bias = False)

        if value == "id":
            self.alpha = nn.Parameter(torch.tensor(alpha)) if alpha is not None else nn.Parameter(torch.randn())
            self.value = None
            self.value_type = "id"
        else:
            self.value = nn.Linear(embed_size, embed_size, bias = False)
            self.alpha = None
            self.value_type = "general"

    def forward(self, x):

        # Compute q, k, v vectors
        queries = self.query(x)
        keys = self.key(x)
        values = self.alpha * x if self.value_type == "id" else self.value(x)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = torch.nn.functional.softmax(self.beta * scores, dim=-1)
        attended = torch.matmul(attention_weights, values)
        
        y = attended + x
        x = y / y.norm(dim=-1, keepdim=True)
        return x

class CustomTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, beta = 1., store_intermediate = False, share = False, value = "general", alpha = None):
        """
        share: if True, all layers have same matrices Q, K, V, if False each layer has its own matrices.
        """
        super(CustomTransformer, self).__init__()
        if not share:
            self.attention_layers = nn.ModuleList([SimpleAttentionIdentity(embed_size, beta = beta, value = value, alpha = alpha) for _ in range(num_layers)])
        else:
            attention_layer = SimpleAttentionIdentity(embed_size, beta = beta, value = value, alpha = alpha)
            self.attention_layers = nn.ModuleList([attention_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.store_intermediate = store_intermediate

    def forward(self, x):
        if self.store_intermediate:
            intermediate = []
            for layer in self.attention_layers:
                x = layer(x)
                intermediate.append(x)
            return intermediate
        for layer in self.attention_layers:
            x = layer(x)
        return x
