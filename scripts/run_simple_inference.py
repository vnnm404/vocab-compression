from tqdm import tqdm

from models.simple import Simple
from utils.tokenizer import gpt2_tokenizer


def main():
    tokenizer = gpt2_tokenizer()

    model = Simple.load_from_checkpoint("checkpoints/simple-proto-epoch=00-val_loss=2.09.ckpt")
    model.eval()

    text = """Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.
    
As Lily was decorating her room, the sky outside became dark. There was a loud"""
    
    for _ in tqdm(range(256)):
        input_ids = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model(**input_ids)
        
        token_id = outputs[0][-1]
        token = tokenizer.decode(token_id)

        text += token

    print(text)


if __name__ == "__main__":
    main()