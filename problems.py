# Loading problems from JSON files
# Grading AI-generated solutions using the criteria

import json
import re
import subprocess
import sys
import tempfile

from llmapi_integration import LLMAPI


def load_math_problems(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        math_problems = json.load(f)

    for problem in math_problems:
        score_sum = sum(criteria["score"] for criteria in problem["criteria"])
        # print(score_sum, item["criteria"])
        assert abs(score_sum - 1.0) < 1e-5, f"Score sum for question '{problem['question'][:10]}...' in '{path}' must be 1"

    return math_problems


def load_writing_problems(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        writing_problems = json.load(f)

    return writing_problems


def load_ps_problems(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        json_list = json.load(f)

    ps_problems = []
    for problem in json_list:
        ps_problems.append({
            'question': create_prompt_from_ps_problem(problem),
            'test_cases': problem.get('test')
        })
    return ps_problems


def create_prompt_from_ps_problem(problem: dict) -> str:
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


def grade_math_problem(math_problem, ai_answer_text, grader: LLMAPI):
    dialogue = build_math_grading_dialogue(math_problem, ai_answer_text)

    # LLMAPI를 사용해 호출
    response = grader.call(dialogue, first_is_system=True)

    # response["response_text"] 에 최종 응답이 들어감
    reply = response.get("response_text", "").strip()
    print(f"Grader LLM reply:\n{reply}\n")

    # TOTAL_SCORE 파싱
    try:
        import re
        match = re.search(r"TOTAL_SCORE\s*=\s*([01](?:\.\d+)?)", reply)
        if match:
            total_score = float(match.group(1))
        else:
            print("No valid TOTAL_SCORE found in grader reply.")
            total_score = 0.0
    except Exception as e:
        print(f"Error parsing TOTAL_SCORE: {e}")
        total_score = 0.0

    return total_score, reply


def build_math_grading_dialogue(math_problem, ai_answer_text) -> list[str]:
    criteria_list = "\n".join(
        [f"{i+1}. {criterion['text']} (배점: {criterion['score']})"
         for i, criterion in enumerate(math_problem["criteria"])]
    )

    system_message = f"""
너는 수학 문제에 대한 AI 답변을 채점하는 전문가야.

문제:
{math_problem['question']}

채점 기준:
{criteria_list}

채점 방법:
각 기준별로 AI 답변이 해당 기준을 충족했는지 서술하시오.
각 기준에 대해 배점을 전부 주거나, 전혀 충족하지 못하면 0점을 부여하시오.
각 기준에 대한 평가와 점수를 반드시 모두 서술하시오.

그 후, 각 기준별 점수를 모두 더한 총합을 구하시오. 총합은 반드시 0.0 이상 1.0 이하의 값이 되어야 한다.

마지막 출력 형식:
- 마지막 줄에 반드시 아래 형식으로 출력할 것 (소문자로 정확히 입력):
TOTAL_SCORE=p

p는 0.0 이상 1.0 이하의 소수 값이다.
TOTAL_SCORE 출력 이후에는 절대 추가 설명을 하지 마시오.
"""

    user_message = f"""
AI가 작성한 답변은 다음과 같습니다:

\"\"\"
{ai_answer_text}
\"\"\"

위 AI 답변을 채점하세요.
위 지침에 따라 반드시 채점 과정을 서술하고, 마지막 줄에 'TOTAL_SCORE=p' 형식으로 최종 점수를 출력하세요.
"""

    dialogue = [system_message.strip(), user_message.strip()]
    return dialogue


def grade_writing_problem(writing_problem, ai_answer_text, grader: LLMAPI):
    dialogue = build_writing_grading_dialogue(writing_problem, ai_answer_text)

    # LLMAPI를 사용해 호출
    response = grader.call(dialogue, first_is_system=True)

    # response["response_text"] 에 최종 응답이 들어감
    reply = response.get("response_text", "").strip()
    print(f"Grader LLM reply:\n{reply}\n")

    # TOTAL_SCORE 파싱
    try:
        import re
        match = re.search(r"TOTAL_SCORE\s*=\s*(\d+)", reply)
        if match:
            raw_score = int(match.group(1))
            # 점수 정규화: 0~1.0
            total_score = raw_score / 100
        else:
            print("No valid TOTAL_SCORE found in grader reply.")
            total_score = 0.0
    except Exception as e:
        print(f"Error parsing TOTAL_SCORE: {e}")
        total_score = 0.0

    return total_score, reply


def build_writing_grading_dialogue(writing_problem, ai_answer_text) -> list[str]:
    system_message = f"""
너는 논술형 문제 채점 전문가야.

주어진 문제에 대한 AI 답변을 채점하시오.

문제:
아래 제시문을 읽고 물음에 답하시오.
{writing_problem['question']}

채점 기준:
{writing_problem['criteria']}

문항 해설:
{writing_problem['commentary']}

출제 의도:
{writing_problem['intention']}

채점 방법:
- 채점 기준과 문항 해설, 출제 의도를 충분히 고려하시오.
- AI 답변이 채점 기준을 얼마나 충실히 충족하는지 평가하시오.
- 반드시 0점 이상 100점 이하의 점수로 채점하시오.
- 채점 과정(기준별 평가 이유)을 반드시 서술하시오.
- 마지막에는 반드시 아래 형식으로 출력하시오 (소문자, 공백 없이 정확히 입력):

TOTAL_SCORE=p

p는 0 이상 100 이하의 정수이다.
TOTAL_SCORE 이후에는 절대 추가 설명을 하지 마시오.
"""

    user_message = f"""
AI가 작성한 답변은 다음과 같습니다:

\"\"\"
{ai_answer_text}
\"\"\"

위 AI 답변을 채점하세요.
위 지침에 따라 채점 과정을 서술하고 마지막 줄에 'TOTAL_SCORE=p' 형식으로 점수를 출력하세요.
"""

    dialogue = [system_message.strip(), user_message.strip()]
    return dialogue


def grade_ps_problem(ps_problem, ai_answer_text):
    try:
        code = extract_python_code(ai_answer_text)
    except ValueError as e:
        print(f"Parsing error: {e}")
        return 0.0, ""

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

    score_percentage = (correct_cases / total_cases) if total_cases > 0 else 0.0
    return score_percentage, ""  # ps grading does not have grader reply


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
    load_math_problems("problems/math.json")

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
