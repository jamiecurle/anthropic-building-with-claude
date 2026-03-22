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
        # V3: + now also specific
        # note: Guidelines
        3: f"""
    Generate a one day meal plan for an athelete that meets their dietary restrictions.
    
    - Height: {prompt_inputs["height"]}
    - Weight: {prompt_inputs["weight"]}
    - Goal: {prompt_inputs["goal"]} 
    - Dietary restrictions: {prompt_inputs["restrictions"]}
    
    Guidelines:
    1. Include accurate daily calorie amount
    2. Show protein, fat, and carb amounts  
    3. Specify when to eat each meal
    4. Use only foods that fit restrictions
    5. List all portion sizes in grams
    6. Keep budget-friendly if mentioned
    """,
        4: f"""
    Generate a one day meal plan for an athelete that meets their dietary restrictions.
    
    - Height: {prompt_inputs["height"]}
    - Weight: {prompt_inputs["weight"]}
    - Goal: {prompt_inputs["goal"]} 
    - Dietary restrictions: {prompt_inputs["restrictions"]}
    
    Follow these steps:
    1. Calulate daily calories needed
    2. Figure out protein, carbs and fat amounts
    3. Plan meal timings around workouts
    4. Choose foods that fit restrictions
    5. Set portion sizes in grams
    6. Adjust for budget if needed
    """,
        5: f"""
    Generate a one day meal plan for an athelete that meets their dietary restrictions.
    
    <athlete_information>
    - Height: {prompt_inputs["height"]}
    - Weight: {prompt_inputs["weight"]}
    - Goal: {prompt_inputs["goal"]} 
    - Dietary restrictions: {prompt_inputs["restrictions"]}
    </athlete_information>
    
    Guidelines:
    1. Include accurate daily calorie amount
    2. Show protein, fat, and carb amounts  
    3. Specify when to eat each meal
    4. Use only foods that fit restrictions
    5. List all portion sizes in grams
    6. Keep budget-friendly if mentioned
    """,
    }

    return PROMPTS[version]


def run_prompt(prompt_inputs: dict[str, Any]) -> str:
    prompt = versioned_prompt(5, prompt_inputs=prompt_inputs)

    # create a list of messages
    messages: list[MessageParam] = []

    # add a user message
    fn.add_user_message(messages, prompt)

    # and now return the chat
    return fn.chat(messages)
