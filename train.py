import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from tokenizer_utils import get_tokenizer
from tinygpt import TinyGPT

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # x is the input sequence
        # y is the target sequence, shifted by 1
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train():
    parser = argparse.ArgumentParser(description="Train TinyGPT")
    parser.add_argument('--tokenizer', type=str, choices=['char', 'bpe', 'unigram'], default='bpe')
    parser.add_argument('--epochs', type=int, default=50)
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
    
    # 2. Prepare dataset and dataloader
    dataset = TextDataset(encoded_text, args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Initialize Model
    model = TinyGPT(vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    loss_history = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for CrossEntropy: (batch_size * seq_len, vocab_size)
            logits = logits.view(-1, tokenizer.vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"--- Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} ---")
        
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
