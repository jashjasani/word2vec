import torchtext
torchtext.disable_torchtext_deprecation_warning()

import os
import random
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
import numpy as np
import h5py
import time
import math

def get_vocab_and_skipgrams(path: str,
                            data_dir:str, 
                            hdf5_chunk_size=(1000000, 2), 
                            skipgram_batch_size=3000000, 
                            chunk_size= 100000, 
                            window_size=2,
                            ns_count=4):
    cwd = os.getcwd()
    # hdf5_path = os.path.join(cwd, data_dir, "skipgram.h5")
    # vocab_path = os.path.join(cwd, data_dir, "vocab.npy")
    tokenizer = get_tokenizer("basic_english", "en")
    vocab = defaultdict(lambda: len(vocab))
    
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    
    # # Initialize HDF5 file for skipgrams
    # with h5py.File(hdf5_path, "w") as hf:
    #     hf.create_dataset('positive_skips', shape=(0, 2), maxshape=(None, 2), 
    #                       dtype='int32', chunks=hdf5_chunk_size)
    
    target_batches = []
    context_batches = []
    label_batches = []
    skipgram_buffer = []
    total_skipgrams = 0
    start_time = time.time()
    
    def process_line(line):
        nonlocal target_batches, context_batches, label_batches, total_skipgrams
        if line.strip():
            seq = [vocab[word] for word in tokenizer(line)]
            targets, contexts, labels = generate_skipgrams(seq, window_size, ns_count, len(vocab))
            target_batches.append(targets)
            context_batches.append(contexts)
            label_batches.append(labels)
            print(targets)
            print(contexts)
            print(labels)
            # if len(skipgram_buffer) >= skipgram_batch_size:
            #     write_skipgrams_to_disk(skipgram_buffer, hdf5_path)
            #     total_skipgrams += len(skipgram_buffer)
            #     skipgram_buffer = []
        return
    
    
    if os.path.isfile(path):
        with open(path, 'r') as file:
            total_lines = sum(1 for _ in file)
            file.seek(0)
            num_chunks = math.ceil(total_lines / chunk_size)
            for chunk in tqdm(iter(lambda: list(islice(file, chunk_size)), []), total=num_chunks, desc="Processing file"):
                for line in chunk:
                    process_line(line)
    else:
        for file in tqdm(list_text_files(path), desc="Processing files"):
            with open(file, 'r') as f:
                for line in f:
                    process_line(line)
    
    # # Write any remaining skipgrams
    # if skipgram_buffer:
    #     write_skipgrams_to_disk(skipgram_buffer, hdf5_path)
    #     total_skipgrams += len(skipgram_buffer)
    
    end_time = time.time()
    print(f"Total skipgrams generated: {total_skipgrams}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Average processing speed: {total_skipgrams / (end_time - start_time):.2f} skipgrams/second")
    
    # Save vocab
    # np.save(vocab_path, dict(vocab))

    return target_batches, context_batches, label_batches, vocab


def generate_skipgrams(sequence, window_size, ns_count, vocab_size):
    targets = []
    contexts = []
    labels = []
    for i, target_word in enumerate(sequence):
        for j in range(max(0, i - window_size), min(len(sequence), i + window_size + 1)):
            if i != j:
                targets.append(target_word)
                contexts.append(sequence[j])
                labels.append(1)
                # yield (target_word, sequence[j], 1)
                negative_word = random.randint(0, vocab_size - 1)
                for _ in range(ns_count):
                    while negative_word in sequence:
                        negative_word = random.randint(0, vocab_size-1)
                    targets.append(target_word)
                    contexts.append(negative_word)
                    labels.append(0)
                    # yield (target_word, negative_word, 0)
    
    return targets, contexts, labels

def write_skipgrams_to_disk(skipgrams, path):
    with h5py.File(path, "a") as hf:
        dataset = hf['positive_skips']
        current_size = dataset.shape[0]
        new_size = current_size + len(skipgrams)
        dataset.resize(new_size, axis=0)
        dataset[current_size:new_size] = skipgrams

def list_text_files(directory):
    return [os.path.join(dirpath, f)
            for dirpath, _, filenames in os.walk(directory)
            for f in filenames if f.endswith('.txt')]

def test():
    get_vocab_and_skipgrams("./shakepere.txt", data_dir="training_data")

if __name__ == "__main__":
    test()