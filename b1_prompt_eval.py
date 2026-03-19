# prompt = f"""
# Please provide a solution to the following task:

# {task}
# """

import json
import datetime
import ast
import re
from statistics import mean
from anthropic.types import MessageParam

import fn


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
        "format" "json" or "python" or "regex",
        "solution_criteria": "Key criteria for evaluating the solution"
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

* Respond only with Python, JSON, or a plain Regex
* Do not add any comments or commentary or explaination
"""
    # now the dance
    messages: list[MessageParam] = []
    fn.add_user_message(messages, prompt)
    fn.add_assistant_message(messages, "```code")
    output = fn.chat(messages, stop_sequences=["```"])
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

    Criteria you should use to evaluate the solution
    <criteria>
    {test_case["solution_criteria"]}
    </criteria>

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


def validate_json(text: str) -> int:
    try:
        json.loads(text)
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text: str) -> int:
    try:
        ast.parse(text)
        return 10
    except SyntaxError:
        return 0


def validate_regex(text: str) -> int:
    try:
        re.compile(text)
        return 10
    except re.error:
        return 0


def grade_syntax(response: str, test_case: dict) -> int:
    match test_case["format"]:
        case "python":
            return validate_python(response)
        case "json":
            return validate_json(response)
        case "regex":
            return validate_regex(response)

    # in prod I'd raise a ValueError here, but as this
    # is learning land, print out something helpful and
    # return zero.
    print("No parseable format found in response")
    return 0


def run_test_case(test_case: dict) -> dict:
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # the grading
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    syntax_score = grade_syntax(output, test_case)

    score = (model_score + syntax_score) / 2

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


def save_results(results: list[dict[str, str | int]]) -> bool:
    try:
        dump = json.dumps(results)
        filename = filename = (
            f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w+") as fp:
            fp.write(dump)
        return True
    except ValueError as e:
        print(e)
        return False
