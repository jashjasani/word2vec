import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torchtext.data.utils import get_tokenizer
import tqdm
from model import SkipGramModel
import json
from nltk.tokenize import word_tokenize



class CustomSkipGramDataset(Dataset):
    def __init__(self, filename, window_size, num_ns):
        self.tokenizer = word_tokenize
        self.vocab = self.create_vocab(filename)
        self.vocab_size = len(self.vocab)
        print("Vocab size = ", self.vocab_size)
        self.window_size = window_size
        self.num_ns = num_ns
        self.sequences = self.load_sequences(filename)
        self.targets, self.contexts, self.labels = self.generate_training_data()

    def create_vocab(self, filename):
        vocab = defaultdict(lambda: len(vocab))
        with open(filename, "r") as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            for line in tqdm.tqdm(f, total=total_lines, desc="Creating vocab"):
                if line.strip():
                    for word in self.tokenizer(line.lower()):
                        vocab[word]
        with open("vocab.json", "w") as f:
            json.dump(vocab, f, ensure_ascii=False)
        return vocab

    def load_sequences(self, filename):
        sequences = []
        with open(filename, "r") as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            for line in tqdm.tqdm(f, total=total_lines, desc="Loading sequences"):
                if line.strip():
                    sequences.append([self.vocab[word] for word in self.tokenizer(line.lower())])
        return sequences

    def generate_training_data(self):
        targets, contexts, labels = [], [], []
        sampling_table = self.make_sampling_table()

        for sequence in tqdm.tqdm(self.sequences, desc="Generating training data"):
            positive_skip_grams = self.skipgrams(sequence, sampling_table)

            for target_word, context_word in positive_skip_grams:
                context_class = torch.tensor([context_word], dtype=torch.long)
                negative_sampling_candidates = self.negative_sampling(context_class)

                context = torch.cat([context_class, negative_sampling_candidates])
                label = torch.tensor([1] + [0] * self.num_ns, dtype=torch.long)

                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

        return targets, contexts, labels

    def make_sampling_table(self):
        return torch.tensor([1.0 / (i + 1)**0.75 if i < self.vocab_size else 0 for i in range(self.vocab_size * 2)])

    def skipgrams(self, sequence, sampling_table):
        positive_skip_grams = []
        for i, target_word in enumerate(sequence):
            for j in range(max(0, i - self.window_size), min(len(sequence), i + self.window_size + 1)):
                if i != j:
                    context_word = sequence[j]
                    if torch.rand(1) < sampling_table[context_word]:
                        positive_skip_grams.append((target_word, context_word))
        return positive_skip_grams

    def negative_sampling(self, true_class):
        num_neg_samples = self.num_ns
        neg_samples = torch.empty(num_neg_samples, dtype=torch.long)
        mask = torch.ones(self.vocab_size, dtype=torch.bool)
        mask[true_class] = False

        for i in range(num_neg_samples):
            while True:
                neg_sample = torch.randint(0, self.vocab_size, (1,))
                if mask[neg_sample]:
                    mask[neg_sample] = False
                    neg_samples[i] = neg_sample
                    break

        return neg_samples

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.targets[index], self.contexts[index], self.labels[index]

def collate_fn(batch):
    targets, contexts, labels = zip(*batch)
    return torch.tensor(targets, dtype=torch.long), torch.stack(contexts), torch.stack(labels).float()

def create_dataloader_skipgram(filename, window_size, num_ns, batch_size, shuffle=True):
    return DataLoader(
        CustomSkipGramDataset(filename, window_size, num_ns),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=5,
        collate_fn=collate_fn
    )

def test():
    
    dl = create_dataloader_skipgram("./shakepere.txt", window_size=2, num_ns=5, batch_size=16, shuffle=False)
    model = SkipGramModel(100,dl.dataset.vocab_size)
    for target, context, label in dl:
        dot = model(target, context)
        print(label)
        loss = torch.nn.CrossEntropyLoss(dot, label)
        print(loss)
        break

if __name__ == "__main__":
    test()