import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from tokenizer_utils import get_tokenizer
from tinygpt import TinyGPT

def train():
    parser = argparse.ArgumentParser(description="Train TinyGPT")
    parser.add_argument('--tokenizer', type=str, choices=['char', 'bpe', 'unigram'], default='bpe')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load tokenizer and data
    vocab_size = 50 if args.tokenizer == 'char' else 300
    tokenizer = get_tokenizer(args.tokenizer, vocab_size=vocab_size)
    
    with open('corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        
    print(f"Tokenizing corpus using {args.tokenizer}...")
    encoded_text = tokenizer.encode(text)
    
    # 2. Prepare dataset function (get_batch)
    def get_batch(batch_size):
        ix = torch.randint(len(encoded_text) - args.seq_len, (batch_size,))
        x = torch.stack([torch.tensor(encoded_text[i:i+args.seq_len], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(encoded_text[i+1:i+args.seq_len+1], dtype=torch.long) for i in ix])
        return x.to(device), y.to(device)
    
    # 3. Initialize Model
    model = TinyGPT(vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    loss_history = []
    
    print("Starting training...")
    model.train()
    for step in range(args.epochs):
        x, y = get_batch(args.batch_size)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Reshape for CrossEntropy: (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, tokenizer.vocab_size)
        y = y.view(-1)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0 or step == args.epochs - 1:
            print(f"Step {step}/{args.epochs} | Loss: {loss.item():.4f}")
            loss_history.append(loss.item())
        
    # 5. Save Loss History and Model Checkpoint
    history_file = f"loss_history_{args.tokenizer}.json"
    with open(history_file, 'w') as f:
        json.dump(loss_history, f)
    print(f"Loss history saved to {history_file}")
    
    checkpoint_file = f"tinygpt_{args.tokenizer}.pt"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Model saved to {checkpoint_file}")

if __name__ == "__main__":
    train()
