import json
import os
from openai import RateLimitError
from tqdm import tqdm
from problems import load_math_problems, load_ps_problems, load_writing_problems
from llmapi_integration import LLMAPI
from conversation import run_llm_conversation
from concurrent.futures import ThreadPoolExecutor, as_completed


# 실험 세팅
DIALOGUE_COUNTS = [1, 2, 4, 8]  # [1, 2, 4, 8, 16]
DOMAIN_ORDER = ["math", "writing", "ps"]
MODEL_PAIRS = [['deepseek', 'deepseek'], ['gpt', 'deepseek'], ['deepseek', 'gpt']]
# Memo: GPT-DeepSeek 대화에서 dialog count가 1일 때, GPT만 1번 답변을 하고 DeepSeek은 답변을 하지 않게 됨.
PROBLEM_FILES = {
    "math": "problems/math.json",
    "writing": "problems/writing.json",
    "ps": "problems/ps.json",
}
MAX_WORKERS = 10

with open('api_key.json', 'r', encoding='utf-8') as f:
    api_keys = json.load(f)

# 결과 저장 경로
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_problem_id(domain, idx):
    return f"{domain}_q{idx + 1}"


def get_save_filepath(domain, idx, model_a_name, model_b_name, num_turns):
    return f"{get_problem_id(domain, idx)}__{model_a_name}_x_{model_b_name}__turns{num_turns}.json"


# 문제 텍스트와 턴 수를 입력받아 초기 프롬프트를 생성하는 함수
def generate_initial_prompt_with_turns(domain: str, problem_text: str, total_turns: int) -> str:
    domain_name_map = {
        "math": "수학",
        "writing": "글쓰기",
        "ps": "PS (Problem Solving)"
    }
    domain_description_map = {
        "math": "한국의 고등학교 수학(수학, 수학 I, 수학 II, 미적분, 확률과 통계, 기하) 범위 내에서 풀이하시오. 결론에는 모든 계산 과정을 포함하라.",
        "writing": "결론에는 글쓰기 평가라 가정하고 논술형 답안의 형식에 맞는 글을 쓰시오.",
        "ps": "결론에는 코드뿐만이 아니라 코드의 핵심 아이디어와, 서로 대화하며 코드를 수정한 방향 또한 포함하라."
    }
    return (
        "너희는 AI끼리 대화하고 있다.\n"
        "AI는 문제를 해결하는 과정에서 실수, 논리적 오류, 잘못된 고정관념이나 편향이 발생할 수 있다.\n"
        "따라서 서로의 답에 잘못된 부분이 있는지 따져가며 교정하여 정확하고 논리적인 답변을 만들어야 한다.\n"
        "AI끼리 대화한다고 해서 문제 풀이를 분담하거나 다른 AI에게 일부 작업을 넘기면 안 된다. 너를 포함한 AI의 각 출력은 문제 전체에 대한 답안을 다루어야 한다.\n"
        f"이번 대화에서는 총 {total_turns}번의 대화 턴(응답)이 주어진다.\n"
        "이때 기회의 수는 두 AI가 같이 사용하는 것으로, 본인이 한 번 응답한 뒤 재응답하면 두 번의 기회가 사용됨을 명심하라.\n"
        "각 AI는 응답할 때마다, 남은 기회가 몇 번인지에 대한 안내 문구를 참고하여 대화하라.\n"
        "만약 '남은 기회: 1번'이라는 안내 문구를 받으면, 반드시 결론을 내야 한다고 선언하고 최종 결론을 작성하라.\n"
        "최종 결론은 '결론:'이라는 문장으로 시작하고, 문제에 대한 최종 답변을 명확하게 제시해야 한다.\n\n"
        "채점자는 결론만 보고 서술형 답안을 채점할 것이므로, 결론에는 대화한 모든 내용을 포함해야 한다."
        "이번에 풀어야 할 문제는 다음과 같다:\n\n"
        f"과목: {domain_name_map.get(domain)}\n\n"
        f"{domain_description_map.get(domain)}"
        "[문제]\n"
        f"{problem_text.strip()}\n\n"
        "대화를 시작하자."
    )


def experiment_one_problem(domain, problem_obj, idx, model_a_name, model_b_name, num_turns):
    problem_id = get_problem_id(domain, idx)
    print(f"\n[Running] {problem_id} | {model_a_name.upper()} x {model_b_name.upper()} | {num_turns} turns")
    save_name = get_save_filepath(domain, idx, model_a_name, model_b_name, num_turns)
    save_path = os.path.join(RESULTS_DIR, save_name)
    if os.path.exists(save_path):
        print(f"Skipping {problem_id} - results already exist")
        return

    # 모델 초기화
    model_a = LLMAPI(model=model_a_name, api_key=api_keys.get(model_a_name))
    model_b = LLMAPI(model=model_b_name, api_key=api_keys.get(model_b_name))

    # 프롬프트 만들기
    prompt = generate_initial_prompt_with_turns(domain, problem_obj['question'], num_turns)

    # 대화 실행
    conversation_log, final_dialogue = run_llm_conversation(
        model_a,
        model_b,
        initial_prompt=prompt,
        num_turns=num_turns
    )
    print(*final_dialogue)

    # 최종 결론 텍스트 (final_dialogue의 마지막 발화 내용 전체 사용)
    final_answer_text = final_dialogue[-1].strip()

    # 채점
    grader = LLMAPI(model='gpt', api_key=api_keys.get('gpt'))  # 채점용 LLM 고정

    if domain == "math":
        from problems import grade_math_problem
        score, grader_reply = grade_math_problem(problem_obj, final_answer_text, grader)
        grading_result = {
            "grader_reply": grader_reply,
            "score": score  # 0.0 ~ 1.0
            # Grader 출력은 grade_math_problem 내부에서 이미 print() 되므로 여기서는 별도 저장 X
        }

    elif domain == "writing":
        from problems import grade_writing_problem
        score, grader_reply = grade_writing_problem(problem_obj, final_answer_text, grader)
        grading_result = {
            "grader_reply": grader_reply,
            "score": score
        }

    elif domain == "ps":
        from problems import grade_ps_problem
        score, grader_reply = grade_ps_problem(problem_obj, final_answer_text)
        grading_result = {
            "grader_reply": grader_reply,
            "score": score
        }

    else:
        raise ValueError(f"Invalid domain: {domain}")

    # 결과 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            "problem_id": problem_id,
            "model_pair": f"{model_a_name}_x_{model_b_name}",
            "num_turns": num_turns,
            "initial_prompt": prompt,
            "conversation_log": conversation_log,
            "grading_result": grading_result
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved result: {save_path}")


def experiment_two_models(futures, domain, problems, model_a_name, model_b_name, num_turns, pbar, executor):
    print(f"\n=== Domain: {domain} ===")
    print(f"\n--- Model: {model_a_name.upper()} x {model_b_name.upper()} ---")
    print(f"\n>>> Dialogue Turns: {num_turns} <<<")

    for idx, problem_obj in enumerate(problems):
        futures.append(
            executor.submit(
                experiment_one_problem,
                domain,
                problem_obj,
                idx,
                model_a_name,
                model_b_name,
                num_turns
            )
        )


# 실험 시작
def experiment():
    problems_dict = {}
    for domain in DOMAIN_ORDER:
        file_path = PROBLEM_FILES[domain]
        if domain == "math":
            problems = load_math_problems(file_path)
        elif domain == "writing":
            problems = load_writing_problems(file_path)
        elif domain == "ps":
            problems = load_ps_problems(file_path)
        else:
            raise ValueError(f"Invalid domain: {domain}. Supported domains are: math, writing, ps")
        problems_dict[domain] = problems

    runs_unit = len(problems_dict) * sum(len(problems) for problems in problems_dict.values())

    # 실험 1: 대화 횟수 비교 (모델 고정 = GPT-GPT)
    total_runs_1 = len(DIALOGUE_COUNTS) * runs_unit
    fixed_model_a = 'gpt'
    fixed_model_b = 'gpt'
    print(f"\n--- Experiment 1: Dialogue Turns Comparison (Model = GPT-GPT) ---")
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            with tqdm(total=total_runs_1) as pbar:
                for num_turns in DIALOGUE_COUNTS:
                    for domain, problems in problems_dict.items():
                        experiment_two_models(futures, domain, problems, fixed_model_a, fixed_model_b, num_turns, pbar, executor)
                for _ in as_completed(futures):
                    pbar.update(1)
    except (RateLimitError, KeyboardInterrupt) as e:
        print(e)

    # 실험 2: 모델 비교 (대화 횟수 고정 = 4턴)
    total_runs_2 = len(MODEL_PAIRS) * runs_unit
    fixed_turns = 4
    print(f"\n--- Experiment 2: Model Comparison (Turns = {fixed_turns}) ---")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        with tqdm(total=total_runs_2) as pbar:
            for model_a_name, model_b_name in MODEL_PAIRS:
                for domain, problems in problems_dict.items():
                    experiment_two_models(futures, domain, problems, model_a_name, model_b_name, fixed_turns, pbar, executor)
            for _ in as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    experiment()
