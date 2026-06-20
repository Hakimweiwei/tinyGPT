import argparse
import torch
import torch.nn.functional as F
from tokenizer_utils import get_tokenizer
from tinygpt import TinyGPT
import os

def generate_text(model, tokenizer, prompt, max_len, device):
    model.eval()
    
    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        encoded_prompt = [tokenizer.pad_id] # fallback if empty
        
    x = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
    
    print(f"Generating from prompt: '{prompt}'")
    
    with torch.no_grad():
        out = model.generate(x, max_len)
            
    generated_ids = out[0].tolist()
    return tokenizer.decode(generated_ids)

def main():
    parser = argparse.ArgumentParser(description="Generate text using TinyGPT")
    parser.add_argument('--prompt', type=str, default="strategi tiktok affiliate yang sukses sangat")
    parser.add_argument('--tokenizer', type=str, choices=['char', 'bpe', 'unigram'], default='bpe')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 50 if args.tokenizer == 'char' else 300
    tokenizer = get_tokenizer(args.tokenizer, vocab_size=vocab_size)
    
    model = TinyGPT(vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len)
    checkpoint_file = f"tinygpt_{args.tokenizer}.pt"
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found. Run train.py first.")
        
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.to(device)
    
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_len, device)
    
    print("\n" + "="*50)
    print(generated_text)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
