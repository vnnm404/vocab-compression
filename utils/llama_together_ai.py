import os
from together import Together
# from dotenv import load_dotenv

# load_dotenv()

model = None


def load_model():
    global model
    model = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    return model


def generate_response(messages):
    load_model()

    completion = model.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        messages=messages,
    )

    return completion.choices[0].message.content


def evaluate_story(story):
    load_model()

    evaluate_prompt = """Given the student's completion of the story after the "***" separator, evaluate their writing holistically, focusing on language skills and creativity. Consider the following aspects in your evaluation:

Language Abilities: Assess grammar, sentence structure, vocabulary usage, and overall coherence. Determine how well the student maintains a natural flow, ensures readability, and uses language effectively to convey ideas.

Creativity: Evaluate the originality of the ideas, how well the student builds on the prescribed beginning, and their ability to craft engaging and imaginative content. Consider how effectively the student incorporates unique or unexpected elements that enhance the narrative.

Provide a concise assessment summarizing the strengths and weaknesses of the student's writing in a single paragraph."""

    messages = [
        {
            "role": "system",
            "content": evaluate_prompt,
        },
        {"role": "user", "content": story},
    ]

    holistic_feedback = generate_response(messages)

    print(holistic_feedback)

    messages.append({"role": "assistant", "content": holistic_feedback})

    grading_prompt = """Your task is to grade the completion based on four criteria and estimate the likely age group of the student. Provide the evaluation in JSON format with the following keys:

1. **'grammar'**: Rate the student's grammar, sentence structure, and language usage on a scale of 1 to 10, where 10 is excellent and 1 is poor.

2. **'creativity'**: Rate the creativity shown in the completion on a scale of 1 to 10, considering the originality of ideas and the student's ability to engage the reader.

3. **'consistency'**: Rate how well the student's completion stays consistent with the beginning of the story on a scale of 1 to 10, considering character, tone, and storyline coherence.

4. **'plot_sense'**: Rate the logical flow and sense of the plot on a scale of 1 to 10, assessing whether the story progression makes sense.

5. **'estimated_age_group'**: Based on the language and content, estimate the age group of the student as one of the following:
   - A: 3 or under
   - B: 4-5
   - C: 6-7
   - D: 8-9
   - E: 10-12
   - F: 13-16

Output your response in the following JSON format without any additional text:

{
  "grammar": <number from 1 to 10>,
  "creativity": <number from 1 to 10>,
  "consistency": <number from 1 to 10>,
  "plot_sense": <number from 1 to 10>,
  "estimated_age_group": "<age group from A to F>"
}"""

    messages.append(
        {
            "role": "user",
            "content": grading_prompt,
        }
    )

    return generate_response(messages)


if __name__ == "__main__":
    # story = "Once upon a time, *** there might was a land far from here, the Popel of this land whwe ghossts :9(9())"
    # story = "Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.\nAs Lily was decorating her room, the sky outside became dark. There was a loud *** clap of thunder that startled her, causing her to drop one of the shiny decorations. To her amazement, instead of breaking, it began to glow softly. Suddenly, all the decorations she had placed started to shimmer and float into the air. The room transformed into a magical wonderland filled with dancing lights and gentle melodies. Lily watched in awe as the decorations painted stories of distant lands on her walls, filling her heart with wonder. That night, she realized that the old house held secrets beyond her imagination, and she felt grateful to be a part of its enchanting world."
    story = """Once upon a time, in an ancient house, there lived a girl named Lily. She loved to decorate her room with pretty things. One day, she found a big box in the attic. She opened it and saw many shiny decorations. Lily was very happy and decided to use them in her room.
    
As Lily was decorating her room, the sky outside became dark. There was a loud *** noise. Lily was scared and wanted to help her mom. She asked her mom if she could help her. 

Her mom said yes and they went to the store. Lily was so happy to have a new friend. She had a new friend and she was happy to have a new friend."""

    response = evaluate_story(story)
    print(response)
