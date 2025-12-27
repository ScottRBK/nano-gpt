import torch 
from torch import Tensor

# define the block size that we are going to be predicting the next token up to
BLOCK_SIZE = 8
# define the batch size, the number of sequences we will process in parallel
BATCH_SIZE = 4

def get_batch(data: Tensor):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) 
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    
    return x, y

def main():

    text = ""
    try:
        with open("input.txt", "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        raise ValueError(f"An error occurred while reading the file. {e}")
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
                
    ## Simple Character Tokenizer - converts individual characters to integers, this is just basic to keep it simple for now
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ["".join([itos[i] for i in l])]
    
    #high dimentional tensor to store all the integers that we have encoded from the input text
    #data.shape = torch.Size[1115394]
    data = torch.tensor(encode(text), dtype=torch.long)

    #split the data into train and validation sets
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # just sets the seed so we can reproduce values - also loving the seed number
    torch.manual_seed(1337)

    xb, yb = get_batch(data=train_data)

    print(f"inputs shape: {xb.shape}, inputs: {xb}") 
    print(f"targets shape: {yb.shape}, targets: {yb}")
    
    for b in range(BATCH_SIZE):
        for t in range(BLOCK_SIZE):
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} then the target is {target}")
    

if __name__ == "__main__":
    main()
