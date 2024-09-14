import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

model = None
tokenizer = None
pipe = None

def load_model():
    global model, tokenizer, pipe
    if model is None or tokenizer is None or pipe is None:
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = model.eval()
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

def unload_model():
    global model, tokenizer, pipe
    del model
    del tokenizer
    del pipe
    model = None
    tokenizer = None
    pipe = None

def generate_response(messages):
    load_model()
    
    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,  # low temperature for consistent evaluations
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = outputs[0]["generated_text"][-1]["content"]
    return response

def evaluate_story(story):
    load_model()

    messages = [
        {
            "role": "system",
            "content": "In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion. Evaluate the student's completion of the story (following the *** separator), focusing on their language abilities and creativity. Provide a concise, holistic evaluation on their writing, without discussing the story's content in at most 1 paragraph. Do not generate a sample completion or provide feedback at this time."
        },
        {
            "role": "user",
            "content": story
        },
    ]

    holistic_feedback = generate_response(messages)

    print(holistic_feedback)

    messages.append({
        "role": "assistant",
        "content": holistic_feedback
    })

    messages.append({
        "role": "user",
        "content": "Now, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning, and whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. Choose from possible age groups:\nA: 3 or under\nB: 4-5\nC: 6-7\nD: 8-9\nE: 10-12\nF: 13-16.\n\nPlease provide the grading in JSON format with the keys 'grammar', 'creativity', 'consistency', 'plot_sense', and 'estimated_age_group'. DO NOT OUTPUT ANYTHING BUT THE JSON."
    })
    
    return generate_response(messages)


if __name__ == "__main__":
    # story = "Once upon a time, *** there might was a land far from here, the Popel of this land whwe ghossts :9(9())"
    story = "Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.\nAs Lily was decorating her room, the sky outside became dark. There was a loud *** clap of thunder that startled her, causing her to drop one of the shiny decorations. To her amazement, instead of breaking, it began to glow softly. Suddenly, all the decorations she had placed started to shimmer and float into the air. The room transformed into a magical wonderland filled with dancing lights and gentle melodies. Lily watched in awe as the decorations painted stories of distant lands on her walls, filling her heart with wonder. That night, she realized that the old house held secrets beyond her imagination, and she felt grateful to be a part of its enchanting world."

    response = evaluate_story(story)
    print(response)
