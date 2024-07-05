from torch import nn
import torch




class SkipGramModel(nn.Module):
    def __init__(self, embed_dim, vocab_size, device):
        super(SkipGramModel, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.device = device
        self.target_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim).to(device)
        self.context_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim).to(device)


    def forward(self, target_word, context_word):
        """
            We find the dot product of target word's embedding and context word's embedding 
            cosine similarity = (u.v) / (||u|| ||v||)
            cosine similarity => 1 vectors are identical 
            cosine similarity => 0 vectors are perpendicular
            cosine similarity => -1 vectors are opposite 
            Our goal here is to maximize the similarity btw target vectors and context vectors
            by maximizing the numerator i.e dot product of u.v
            The log sigmoid of dot product gives Prob(Wc|Wt)
        """
        if len(target_word.shape) == 2:
            target = target.squeeze(1)
        t_embed = self.target_embedding(target_word)
        c_embed = self.context_embedding(context_word)
        dot = torch.bmm(t_embed.unsqueeze(1), c_embed.transpose(1,2)).squeeze(1)
        return dot