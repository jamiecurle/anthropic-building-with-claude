from typing import Any
from anthropic.types import MessageParam
from functools import cache

import fn


@cache
def get_evaluator() -> fn.PromptEvaluator:
    evaluator = fn.PromptEvaluator()
    return evaluator


def generate_dataset() -> dict[str, Any]:
    # we need an evaluator
    evaluator = get_evaluator()

    # now a dataset
    dataset = evaluator.generate_dataset(
        # purpose or goal of the prompt
        task_description="Write a compact concise 1 day meal plan for a single athlete",
        # Describe the different inputs that your prompt requires
        prompt_inputs_spec={
            "height": "Athletes height in cm",
            "weight": "Atheletes weight in kg",
            "goal": "Goal of the athelete",
            "restrictions": "Dietary restrictions of the athlete",
        },
        # file to store dataset
        output_file="b2_dataset.json",
        # finally the number of test cases.num_cases
        # in a prod situation you'd want 50+
        num_cases=3,
    )
    return dataset


def run_evaluator() -> list:
    # we need an evaluator
    evaluator = get_evaluator()

    # now run the results
    results: list = evaluator.run_evaluation(
        run_prompt,
        dataset_file="b2_dataset.json",
        extra_criteria="""
        The output should include:
        - Daily caloric total
        - Macronutrient breakdown
        - Meals with exact foods, portions and timing.
        """,
    )
    return results


def versioned_prompt(version: int, prompt_inputs: dict[str, str]) -> str:
    PROMPTS = {
        # starter prompt ... 2.0 score
        1: f"""
    What should this person eat?
    
    - Height: {prompt_inputs["height"]}
    - Weight: {prompt_inputs["weight"]}
    - Goal: {prompt_inputs["goal"]} 
    - Dietary restrictions: {prompt_inputs["restrictions"]}
    """,
        # V2: clear and direct....
        # note: action verb at the start along with a clear task.
        2: f"""
    Generate a one day meal plan for an athelete that meets their dietary restrictions.
    
    - Height: {prompt_inputs["height"]}
    - Weight: {prompt_inputs["weight"]}
    - Goal: {prompt_inputs["goal"]} 
    - Dietary restrictions: {prompt_inputs["restrictions"]}
    """,
    }

    return PROMPTS[version]


def run_prompt(prompt_inputs: dict[str, Any]) -> str:
    prompt = versioned_prompt(2, prompt_inputs=prompt_inputs)

    # create a list of messages
    messages: list[MessageParam] = []

    # add a user message
    fn.add_user_message(messages, prompt)

    # and now return the chat
    return fn.chat(messages)
