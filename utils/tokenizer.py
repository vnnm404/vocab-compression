from transformers import AutoTokenizer

def gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    tokenizer.model_max_length = 1_000_000  # disable warnings
    return tokenizer

def main():
    tokenizer = gpt2_tokenizer()
    print(tokenizer)
    print(len(tokenizer))
    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.pad_token, tokenizer.eos_token)

if __name__ == "__main__":
    main()
