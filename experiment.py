import json
import os
from problems import load_math_problems, load_ps_problems, load_writing_problems
from llmapi_integration import LLMAPI
from conversation import run_llm_conversation


# 실험 세팅
MODEL_PAIRS = [['gpt', 'gpt'], ['gpt', 'deepseek'], ['deepseek', 'gpt'], ['deepseek', 'deepseek']]
# Memo: GPT-DeepSeek 대화에서 dialog count가 1일 때, GPT만 1번 답변을 하고 DeepSeek은 답변을 하지 않게 됨.
DIALOGUE_COUNTS = [4]  # [1, 2, 4, 8, 16]
PROBLEM_FILES = {
    "writing": "problems/writing.json",
    "math": "problems/math.json",
    "ps": "problems/ps.json",
}

with open('api_key.json', 'r', encoding='utf-8') as f:
    api_keys = json.load(f)

# 결과 저장 경로
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# 문제 텍스트와 턴 수를 입력받아 초기 프롬프트를 생성하는 함수
def generate_initial_prompt_with_turns(domain: str, problem_text: str, total_turns: int) -> str:
    domain_name_map = {
        "math": "수학",
        "writing": "글쓰기",
        "ps": "PS (Problem Solving)"
    }
    domain_name = domain_name_map.get(domain, domain)
    template = (
        "너희는 AI끼리 대화하고 있다.\n"
        "AI는 문제를 해결하는 과정에서 실수, 논리적 오류, 잘못된 고정관념이나 편향이 발생할 수 있다.\n"
        "따라서 너희는 서로 협력하여 대화하면서 이러한 오류나 편향을 발견하고 교정하는 것이 목적이다.\n"
        "최종적으로는 정확하고 논리적인 최선의 답변을 함께 만들어내야 한다.\n\n"
        "이번 대화에서는 총 {total_turns}번의 대화 턴(응답)이 주어진다.\n"
        "이때 기회의 수는 두 AI가 같이 사용하는 것으로, 본인이 한 번 응답한 뒤 재응답하면 두 번의 기회가 사용됨을 명심하라.\n"
        "각 AI는 응답할 때마다, 남은 기회가 몇 번인지에 대한 안내 문구를 참고하여 대화하라.\n"
        "만약 '남은 기회: 1번'이라는 안내 문구를 받으면, 반드시 결론을 내야 한다고 선언하고 최종 결론을 작성하라.\n"
        "최종 결론은 '결론:'이라는 문장으로 시작하고, 문제에 대한 최종 답변을 명확하게 제시해야 한다.\n\n"
        "이번에 풀어야 할 문제는 다음과 같다:\n\n"
        "과목: {domain_name}\n\n"
        "[문제]\n"
        "{problem_text}\n\n"
        "대화를 시작하자."
    )

    prompt = template.format(
        total_turns=total_turns,
        domain_name=domain_name,
        problem_text=problem_text.strip()
    )

    return prompt


def experiment_two_models(domain, problems, model_a_name, model_b_name, num_turns):
    print(f"\n--- Model: {model_a_name.upper()} x {model_b_name.upper()} ---")
    print(f"\n>>> Dialogue Turns: {num_turns} <<<")

    # 모델 초기화
    model_a = LLMAPI(model=model_a_name, api_key=api_keys.get(model_a_name))
    model_b = LLMAPI(model=model_b_name, api_key=api_keys.get(model_b_name))

    for idx, problem_obj in enumerate(problems):
        # 프롬프트 만들기
        prompt = generate_initial_prompt_with_turns(domain, problem_obj['question'], num_turns)
        problem_id = f"{domain}_q{idx + 1}"

        print(f"\n[Running] {problem_id} | {model_a_name.upper()} x {model_b_name.upper()} | {num_turns} turns")

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
        save_name = f"{problem_id}__{model_a_name}_x_{model_b_name}__turns{num_turns}.json"
        save_path = os.path.join(RESULTS_DIR, save_name)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                "problem_id": problem_id,
                "model_pair": f"{model_a_name}_x_{model_b_name}",
                "num_turns": num_turns,
                "initial_prompt": prompt,
                "conversation_log": conversation_log,
                "final_answer_text": final_answer_text,
                "grading_result": grading_result
            }, f, indent=2, ensure_ascii=False)

        print(f"Saved result: {save_path}")


# 실험 시작
def experiment():
    for domain, file_path in PROBLEM_FILES.items():
        print(f"\n=== Domain: {domain} ===")
        if domain == "math":
            problems = load_math_problems(file_path)
        elif domain == "writing":
            problems = load_writing_problems(file_path)
        elif domain == "ps":
            problems = load_ps_problems(file_path)
        else:
            raise ValueError(f"Invalid domain: {domain}. Supported domains are: math, writing, ps")

        # 실험 1: 모델 비교 (대화 횟수 고정 = 4턴)
        fixed_turns = 4
        print(f"\n--- Experiment 1: Model Comparison (Turns = {fixed_turns}) ---")
        for model_a_name, model_b_name in MODEL_PAIRS:
            experiment_two_models(domain, problems, model_a_name, model_b_name, num_turns=fixed_turns)

            # Pause 기능
            user_input = input("Press Enter to continue to the next model pair, or type 'pause' to pause: ")
            if user_input.lower() == 'pause':
                input("Paused. Press Enter to resume.")

        # 실험 2: 대화 횟수 비교 (모델 고정 = GPT-GPT)
        fixed_model_a = 'gpt'
        fixed_model_b = 'gpt'
        print(f"\n--- Experiment 2: Dialogue Turns Comparison (Model = GPT-GPT) ---")
        for num_turns in DIALOGUE_COUNTS:
            experiment_two_models(domain, problems, fixed_model_a, fixed_model_b, num_turns=num_turns)

            # Pause 기능
            user_input = input("Press Enter to continue to the next dialogue turns, or type 'pause' to pause: ")
            if user_input.lower() == 'pause':
                input("Paused. Press Enter to resume.")


if __name__ == "__main__":
    experiment()
