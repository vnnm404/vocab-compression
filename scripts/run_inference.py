from tqdm import tqdm

from models.gpt2 import GPT2
from utils.tokenizer import gpt2_tokenizer


def main():
    tokenizer = gpt2_tokenizer()

    model = GPT2.load_from_checkpoint("checkpoints/gpt2-epoch=00-val_loss=1.22.ckpt")
    model.eval()

    text = """Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.
    
As Lily was decorating her room, the sky outside became dark. There was a loud"""
    
    # for _ in tqdm(range(256)):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    outputs = model.test_step({"input_ids": input_ids}, 0)

    print(outputs[0])


if __name__ == "__main__":
    main()
