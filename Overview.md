## Break Down of Code
### `main.py`
1. Import a File containing all our data that we want to train on
1. Create a volacbolary of all the unique characters in that file
1. Create a simple character tokenizer, where we basic convert a character to an integer value.
1. We then encode the text using the tokenizer and store it in a tensor 
1. Split the data into train and validation
1. We then define the `block_size` - this is the context length in which we will predict the next token from (worth nothing
that we will train up to the block size as to ensure we expose the transformer to different sample sizes)
1. We define the `batch_size` which is the number of sequneces we will process in parallel
1. We then use the `get_batch` function to retrieve a batch of training data
The key thing to observe is that we are now seeing a batch - the batch dimeons of 4, each containing sepearte blocks of 8 tokens - the time dimension 

if we print `line 48`:
```
inputs shape: torch.Size([4, 8]), inputs: tensor([[24, 43, 58,  5, 57,  1, 46, 43],
        [44, 53, 56,  1, 58, 46, 39, 58],
        [52, 58,  1, 58, 46, 39, 58,  1],
        [25, 17, 27, 10,  0, 21,  1, 54]])
targets shape: torch.Size([4, 8]), targets: tensor([[43, 58,  5, 57,  1, 46, 43, 39],
        [53, 56,  1, 58, 46, 39, 58,  1],
        [58,  1, 58, 46, 39, 58,  1, 46],
        [17, 27, 10,  0, 21,  1, 54, 39]])
```
<b>Note:</b> it is interesting when executing the below..
```python
    for b in range(BATCH_SIZE):
        for t in range(BLOCK_SIZE):
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} then the target is {target}")
```
You get this output, which demonstrates that we are going to learn various combination of token lengths, not the full block size,
this is a really nice approach.
```
when input is [24] then the target is 43
when input is [24, 43] then the target is 58
when input is [24, 43, 58] then the target is 5
when input is [24, 43, 58, 5] then the target is 57
when input is [24, 43, 58, 5, 57] then the target is 1
when input is [24, 43, 58, 5, 57, 1] then the target is 46
when input is [24, 43, 58, 5, 57, 1, 46] then the target is 43
when input is [24, 43, 58, 5, 57, 1, 46, 43] then the target is 39
when input is [44] then the target is 53
...
```