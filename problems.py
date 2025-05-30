# Loading problems from JSON files
# Grading AI-generated solutions using the criteria

import json
import re
import subprocess
import sys
import tempfile


def load_exam_problems(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        exam_problems = json.load(f)

    for item in exam_problems:
        score_sum = sum(criteria["score"] for criteria in item["criteria"])
        # print(score_sum, item["criteria"])
        assert abs(score_sum - 1.0) < 1e-5, f"Score sum for question '{item['question']}' in '{path}' must be 1"

    return exam_problems


def load_ps_problems(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        json_list = json.load(f)

    ps_problems = []
    for problem in json_list:
        ps_problems.append({
            'question': create_prompt_from_problem(problem),
            'test_cases': problem.get('test')
        })
    return ps_problems


def create_prompt_from_problem(problem: dict) -> str:
    # Extract problem details
    time_limit = problem.get('timeLimit', '')
    memory_limit = problem.get('memoryLimit', '')
    question = problem.get('question', '')
    input_desc = problem.get('input', '')
    output_desc = problem.get('output', '')
    constraints = problem.get('constraints', [])

    # Format constraints as a string
    constraints_str = '\n'.join([f"- {constraint}" for constraint in constraints])

    # Create the prompt
    prompt = f"""아래에 주어지는 PS 문제를 해결하시오. Python으로 코드를 작성하시오.
시간 제한: {time_limit}
메모리 제한: {memory_limit}

문제:
{question}

입력:
{input_desc}

출력:
{output_desc}

제한사항:
{constraints_str}"""

    return prompt


def grade_exam_problem(exam_problem):
    ...


def grade_ps_problem(ps_problem, ai_answer_text):
    try:
        code = extract_python_code(ai_answer_text)
    except ValueError as e:
        print(f"Parsing error: {e}")
        return 0.0

    test_cases = ps_problem.get('testCases', [])
    total_cases = len(test_cases)
    correct_cases = 0

    for i, test_case in enumerate(test_cases):
        input_data = test_case['input']
        expected_output = test_case['output'].strip()

        actual_output = run_python_code(code, input_data).strip()

        print(f"Test {i+1}:")
        print(f"Input:\n{input_data}")
        print(f"Expected Output:\n{expected_output}")
        print(f"Actual Output:\n{actual_output}\n")

        if actual_output == expected_output:
            correct_cases += 1

    score_percentage = (correct_cases / total_cases) * 100 if total_cases > 0 else 0.0
    return score_percentage

def extract_python_code(answer_text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, answer_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("No Python code block found in the answer.")

def run_python_code(code: str, input_str: str, timeout: float = 2.0) -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_filename = tmp_file.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_filename],
            input=input_str.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        output = result.stdout.decode('utf-8').strip()
        return output
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)}"


if __name__ == '__main__':
    # Load exam problems
    load_exam_problems("problems/math.json")

    # Load and process programming problems
    problem_mappings = load_ps_problems("problems/ps.json")

    # Print example to verify
    if problem_mappings:
        print(f"Total problems processed: {len(problem_mappings)}")
        print("\nExample prompt:")
        print(problem_mappings[0]['question'])
        print("\nExample test cases:")
        for i, test_case in enumerate(problem_mappings[0]['test_cases'][:2]):
            print(f"Test {i+1}:")
            print(f"Input: {test_case['input']}")
            print(f"Output: {test_case['output']}")
