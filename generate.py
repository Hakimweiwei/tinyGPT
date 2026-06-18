import argparse
import torch
import torch.nn.functional as F
from tokenizer_utils import get_tokenizer
from tinygpt import TinyGPT
import os

def generate_text(model, tokenizer, prompt, max_len, temperature, top_k, device):
    model.eval()
    
    # Encode prompt
    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        encoded_prompt = [tokenizer.pad_id] # fallback if empty
        
    x = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
    
    print(f"Generating from prompt: '{prompt}'")
    
    with torch.no_grad():
        for _ in range(max_len):
            # Crop to max_seq_len to avoid index out of bounds in positional encoding
            seq_len = x.size(1)
            input_x = x[:, -model.max_seq_len:]
            
            logits = model(input_x)
            # Take the logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
            
            # Stop if EOS token is generated (optional, depending on tokenizer)
            # if next_token.item() == tokenizer.sp.eos_id(): break
            
    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)

def main():
    parser = argparse.ArgumentParser(description="Generate text using TinyGPT")
    parser.add_argument('--prompt', type=str, default="strategi sukses menjadi tiktok affiliate adalah")
    parser.add_argument('--tokenizer', type=str, choices=['char', 'bpe', 'unigram'], default='bpe')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=10)
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
    
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_len, args.temperature, args.top_k, device)
    
    print("\n" + "="*50)
    print(generated_text)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
