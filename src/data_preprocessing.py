import numpy as np
from datasets import load_dataset
import os

def load_and_preprocess_data(max_len=128, vocab_size=15000):
    """
    Loads gtfintechlab/finer-ord-bio from Hugging Face, builds vocabulary,
    pads sequences using native Python padding (token padding: 0, tag padding: -1),
    and returns tokenized train, validation, and test datasets.
    """
    print("Loading gtfintechlab/finer-ord-bio dataset from Hugging Face...")
    dataset = load_dataset("gtfintechlab/finer-ord-bio")
    
    # Extract training data to build vocabulary
    train_tokens = dataset["train"]["tokens"]
    train_tags = dataset["train"]["tags"]
    
    # Define BIO labels list corresponding to indices [0, 1, 2, 3, 4, 5, 6]
    label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    print(f"Entities found: {label_list}")
    num_classes = len(label_list)
    
    # Build word vocabulary
    word_counts = {}
    for sentence in train_tokens:
        for word in sentence:
            if word is not None:
                w = word.lower()
                word_counts[w] = word_counts.get(w, 0) + 1
            
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Keep top (vocab_size - 2) words to leave room for PAD (0) and UNK (1)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in sorted_words[:vocab_size - 2]:
        vocab[word] = len(vocab)
        
    print(f"Vocabulary size: {len(vocab)}")
    
    # Helper to convert sentence to list of word indices
    def sentence_to_ids(tokens):
        return [vocab.get(t.lower() if t is not None else "<UNK>", 1) for t in tokens]
        
    # Python native helper to pad sequences
    def pad_sequence(seq, max_length, pad_value):
        if len(seq) < max_length:
            return seq + [pad_value] * (max_length - len(seq))
        else:
            return seq[:max_length]
            
    # Process each split
    def process_split(split_name):
        tokens_split = dataset[split_name]["tokens"]
        tags_split = dataset[split_name]["tags"]
        
        # Map tokens to IDs
        x_ids = [sentence_to_ids(sent) for sent in tokens_split]
        
        # Pad token sequences with 0 (PAD)
        x_pad = np.array([pad_sequence(sent, max_len, 0) for sent in x_ids], dtype=np.int64)
        
        # Pad tag sequences with -1 (used in CrossEntropyLoss to ignore loss)
        y_pad = np.array([pad_sequence(tags, max_len, -1) for tags in tags_split], dtype=np.int64)
        
        return x_pad, y_pad, tokens_split, tags_split
        
    print("Preprocessing train set...")
    x_train, y_train, train_raw_tokens, train_raw_tags = process_split("train")
    
    print("Preprocessing validation set...")
    x_val, y_val, val_raw_tokens, val_raw_tags = process_split("validation")
    
    print("Preprocessing test set...")
    x_test, y_test, test_raw_tokens, test_raw_tags = process_split("test")
    
    # Pack metadata
    metadata = {
        "vocab": vocab,
        "label_list": label_list,
        "num_classes": num_classes,
        "vocab_size": len(vocab),
        "max_len": max_len
    }
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), metadata
