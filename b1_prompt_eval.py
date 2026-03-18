# prompt = f"""
# Please provide a solution to the following task:

# {task}
# """
import json
import fn
from statistics import mean
from anthropic.types import MessageParam


def generate_dataset() -> dict:
    prompt = """
Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
    {
        "task": "Description of task",
    },
    ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""
    # create a list of messages
    messages: list[MessageParam] = []

    # make a user message
    fn.add_user_message(messages, prompt)

    # make an assistant message
    fn.add_assistant_message(messages, "```json")

    # now do that chat with a stop sequence
    response = fn.chat(messages, stop_sequences=["```"])

    fn.inspect(response)

    # now return json
    return json.loads(response)


def save_dataset(dataset: dict) -> bool:
    with open("b_dataset.json", "w+") as fp:
        fp.write(json.dumps(dataset, indent=2))

    return True


def open_dataset() -> dict:
    with open("b_dataset.json", "r+") as fp:
        dataset = json.load(fp)
    return dataset


def run_prompt(test_case: dict) -> str:
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""

    messages: list[MessageParam] = []
    fn.add_user_message(messages, prompt)
    output = fn.chat(messages)
    return output


def grade_by_model(test_case: dict, output: str) -> dict:
    eval_prompt = f"""
    You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

    Original Task:
    <task>
    {test_case["task"]}
    </task>

    Solution to Evaluate:
    <solution>
    {output}
    </solution>

    Output Format
    Provide your evaluation as a structured JSON object with the following fields, in this specific order:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement
    - "reasoning": A concise explanation of your overall assessment
    - "score": A number between 1-10

    Respond with JSON. Keep your response concise and direct.
    Example response shape:
    {{
        "strengths": string[],
        "weaknesses": string[],
        "reasoning": string,
        "score": number
    }}
        """

    messages = []
    fn.add_user_message(messages, eval_prompt)
    fn.add_assistant_message(messages, "```json")

    eval_text = fn.chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


def run_test_case(test_case: dict) -> dict:
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # TODO - Grading
    model_grade = grade_by_model(test_case, output)
    score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    return {
        "reasoning": reasoning,
        "output": output,
        "test_case": test_case,
        "score": score,
    }


def run_eval(dataset: dict) -> list:
    """Loads the dataset and calls run_test_case with each case"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"Average_score: {average_score}")

    return results
