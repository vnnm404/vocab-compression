import random
import json
import torch
from tqdm import tqdm
from utils.llama_together_ai import evaluate_story
from utils.tokenizer import gpt2_tokenizer
from models.gpt2 import GPT2

# Mapping for age groups
AGE_GROUP_MAP = {"A": 1.5, "B": 4.5, "C": 6.5, "D": 8.5, "E": 11, "F": 14.5}

def evaluate_model(model, num_stories=10, num_repeats=2):
    """
    Evaluate the model on a sample of stories by generating continuations and scoring them.

    Args:
        model: The model to be evaluated.
        num_stories: Number of stories to sample for evaluation.
        num_repeats: Number of times to repeat evaluation for each story.

    Returns:
        A dictionary with the average evaluation metrics.
    """
    with open("data/50_stories.json", "r") as file:
        stories = json.load(file)

    # Randomly sample a subset of stories
    sampled_stories = random.sample(stories, num_stories)

    # Initialize average metric accumulators
    total_grammar, total_creativity, total_consistency, total_plot_sense, total_age = 0, 0, 0, 0, 0

    # Loop through each story and evaluate
    for original_story in tqdm(sampled_stories):
        for _ in range(num_repeats):
            # Use a random portion of the original story
            truncated_story = original_story[:random.randint(4, len(original_story) // 2)]

            # Encode the story using the model's tokenizer
            input_ids = model.tokenizer.encode(truncated_story, return_tensors="pt").to(model.device)

            # Generate a continuation of the story
            generated_story = model.test_step({"input_ids": input_ids}, 0)
            full_story = truncated_story + " ***" + generated_story[0][len(truncated_story):]

            # Evaluate the generated story
            eval_message = evaluate_story(full_story)
            evaluations = json.loads(eval_message)

            # Accumulate the evaluation metrics
            total_grammar += evaluations["grammar"]
            total_creativity += evaluations["creativity"]
            total_consistency += evaluations["consistency"]
            total_plot_sense += evaluations["plot_sense"]
            total_age += AGE_GROUP_MAP[evaluations["estimated_age_group"]]

    # Calculate averages
    num_evaluations = num_stories * num_repeats
    avg_grammar = total_grammar / num_evaluations
    avg_creativity = total_creativity / num_evaluations
    avg_consistency = total_consistency / num_evaluations
    avg_plot_sense = total_plot_sense / num_evaluations
    avg_age = total_age / num_evaluations

    return {
        "avg_grammar": avg_grammar,
        "avg_creativity": avg_creativity,
        "avg_consistency": avg_consistency,
        "avg_plot_sense": avg_plot_sense,
        "avg_age": avg_age,
    }

def main():
    """
    Main function to load the model and evaluate it on the provided stories.
    """
    # Initialize tokenizer and load the model from a checkpoint
    tokenizer = gpt2_tokenizer()
    model = GPT2.load_from_checkpoint(
        "checkpoints/gpt2-eos-epoch=00-val_loss=1.40.ckpt", tokenizer=tokenizer
    )
    model.eval()

    # Evaluate the model and print the results
    evaluation_results = evaluate_model(model)
    print(evaluation_results)

if __name__ == "__main__":
    main()
