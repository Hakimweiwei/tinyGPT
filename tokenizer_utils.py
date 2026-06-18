import sentencepiece as spm
import os

class TokenizerWrapper:
    def __init__(self, model_prefix):
        self.sp = spm.SentencePieceProcessor()
        self.model_prefix = model_prefix
        
    def load(self):
        model_file = f"{self.model_prefix}.model"
        if os.path.exists(model_file):
            self.sp.load(model_file)
            return True
        return False
        
    def encode(self, text):
        return self.sp.encode_as_ids(text)
        
    def decode(self, ids):
        return self.sp.decode_ids(ids)
        
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
        
    @property
    def pad_id(self):
        return self.sp.pad_id()
        
def train_tokenizer(input_file, model_prefix, model_type="bpe", vocab_size=500):
    """
    Train a sentencepiece tokenizer.
    model_type can be: 'bpe', 'unigram', 'char', 'word'
    """
    print(f"Training {model_type} tokenizer...")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]'
    )
    print(f"Tokenizer {model_prefix} trained successfully.")
    
def get_tokenizer(tokenizer_type, input_file="corpus.txt", vocab_size=500):
    prefix = f"tokenizer_{tokenizer_type}"
    tokenizer = TokenizerWrapper(prefix)
    
    if not tokenizer.load():
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{input_file} not found. Run prepare_corpus.py first.")
        train_tokenizer(input_file, prefix, tokenizer_type, vocab_size)
        tokenizer.load()
        
    return tokenizer

if __name__ == "__main__":
    # Test training and loading all three as per assignment requirement
    if os.path.exists("corpus.txt"):
        get_tokenizer("char", vocab_size=50) # characters only need small vocab
        get_tokenizer("bpe", vocab_size=120)
        get_tokenizer("unigram", vocab_size=100)
        print("All tokenizers ready.")
    else:
        print("corpus.txt is missing!")
