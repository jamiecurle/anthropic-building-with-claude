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
        # XML tags to provide extre context
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
        #
        # ONESHOT/MULTISHOT
        # Below we've taken some of the data from the eveal report and given it as a "one shot"
        # example of what the sample input and ideal output is.  If we'd have given more examples
        # it would be considered to be multishot.
        6: f"""
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
        6: f"""
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
    
    Here is an example with a sample input and an ideal output:
    <sample_input>
    height: 180
    weight: 85
    goal: Build muscle and strength through weightlifting with high protein intake
    restrictions: None
    </sample_input>
    <ideal_output>
    # One-Day Muscle-Building Meal Plan
    **Athlete: 180cm, 85kg Male Weightlifter**

    ---

    ## Daily Targets
    - **Calories:** 3,200 kcal
    - **Protein:** 170g (2.0g per kg bodyweight)
    - **Carbs:** 400g
    - **Fat:** 85g

    ---

    ## BREAKFAST (7:00 AM)
    **Oatmeal Power Bowl**
    - Rolled oats: 80g
    - Whole milk: 250ml
    - Banana: 150g
    - Peanut butter: 30g
    - Honey: 15g

    | Macro | Amount |
    |-------|--------|
    | Calories | 650 kcal |
    | Protein | 18g |
    | Carbs | 85g |
    | Fat | 20g |

    ---

    ## MID-MORNING SNACK (10:00 AM)
    **Protein Shake**
    - Whey protein powder: 30g
    - Greek yogurt: 150g
    - Berries (frozen): 100g
    - Water: 200ml

    | Macro | Amount |
    |-------|--------|
    | Calories | 280 kcal |
    | Protein | 35g |
    | Carbs | 20g |
    | Fat | 3g |

    ---

    ## LUNCH (1:00 PM)
    **Grilled Chicken & Rice**
    - Chicken breast: 200g
    - Brown rice (cooked): 250g
    - Broccoli: 150g
    - Olive oil: 10ml

    | Macro | Amount |
    |-------|--------|
    | Calories | 750 kcal |
    | Protein | 52g |
    | Carbs | 85g |
    | Fat | 12g |

    ---

    ## PRE-WORKOUT SNACK (3:30 PM)
    **Quick Carbs + Protein**
    - White bread: 80g
    - Turkey breast: 100g
    - Jam: 20g

    | Macro | Amount |
    |-------|--------|
    | Calories | 380 kcal |
    | Protein | 28g |
    | Carbs | 50g |
    | Fat | 4g |

    ---

    ## DINNER (7:00 PM)
    **Lean Beef & Sweet Potato**
    - Ground beef (90% lean): 180g
    - Sweet potato: 250g
    - Mixed vegetables: 150g
    - Olive oil: 10ml

    | Macro | Amount |
    |-------|--------|
    | Calories | 650 kcal |
    | Protein | 37g |
    | Carbs | 75g |
    | Fat | 18g |

    ---

    ## EVENING SNACK (9:30 PM)
    **Casein Protein**
    - Cottage cheese: 200g
    - Almonds: 25g

    | Macro | Amount |
    |-------|--------|
    | Calories | 490 kcal |
    | Protein | 30g |
    | Carbs | 10g |
    | Fat | 28g |

    ---

    ## Daily Totals
    | Macro | Target | Actual |
    |-------|--------|--------|
    | **Calories** | 3,200 | 3,200 |
    | **Protein** | 170g | 170g |
    | **Carbs** | 400g | 400g |
    | **Fat** | 85g | 85g |

    ---

    ## Notes
    ✓ High protein spread throughout the day for optimal muscle synthesis  
    ✓ Carbs timed around workout for energy and recovery  
    ✓ Budget-friendly staple foods  
    ✓ Adjust portions based on actual training intensity
    </ideal_output>
    
    The solution comprehensively meets all mandatory requirements: includes daily caloric total, complete macronutrient breakdown, 6 meals with exact foods and portions with specific timing. All secondary criteria are satisfied and caloric total appropriately supports the athletes. The only minor gap is the absence of explicit post-workout meal timing recommendations. 
    
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
