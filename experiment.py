import json
import os
from problems import load_exam_problems, load_ps_problems
from llmapi_integration import LLMAPI
from conversation import run_llm_conversation


# 실험 세팅
MODEL_PAIRS = [['gpt', 'gpt'], ['gpt', 'deepseek'], ['deepseek', 'deepseek']]
# Memo: GPT-DeepSeek 대화에서 dialog count가 1일 때, GPT만 1번 답변을 하고 DeepSeek은 답변을 하지 않게 됨.
DIALOGUE_COUNTS = [4]  # [1, 2, 4, 8, 16]
PROBLEM_FILES = {
    "math": "problems/math.json",
    "writing": "problems/writing.json",
    "ps": "problems/ps.json"
}

with open('api_key.json', 'r', encoding='utf-8') as f:
    api_keys = json.load(f)

# 결과 저장 경로
RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# 문제 텍스트와 턴 수를 입력받아 초기 프롬프트를 생성하는 함수
def generate_initial_prompt_with_turns(problem_text: str, total_turns: int) -> str:
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
        "[문제]\n"
        "{problem_text}\n\n"
        "대화를 시작하자."
    )

    prompt = template.format(
        total_turns=total_turns,
        problem_text=problem_text.strip()
    )

    return prompt


# 실험 시작
def experiment():
    for domain, file_path in PROBLEM_FILES.items():
        print(f"\n=== Domain: {domain} ===")
        problems = load_ps_problems(file_path) if domain == "ps" else load_exam_problems(file_path)

        for model_a_name, model_b_name in MODEL_PAIRS:
            print(f"\n--- Model: {model_a_name.upper()} x {model_b_name.upper()} ---")

            # 모델 초기화
            model_a = LLMAPI(model=model_a_name, api_key=api_keys.get(model_a_name))
            model_b = LLMAPI(model=model_b_name, api_key=api_keys.get(model_b_name))
            for num_turns in DIALOGUE_COUNTS:
                print(f"\n>>> Dialogue Turns: {num_turns} <<<")

                for idx, problem_obj in enumerate(problems):
                    # 프롬프트 만들기
                    prompt = generate_initial_prompt_with_turns(problem_obj['question'], num_turns)
                    problem_id = f"{domain}_q{idx + 1}"

                    print(f"\n[Running] {problem_id} | {model_a_name.upper()} x {model_b_name.upper()} | {num_turns} turns")

                    # 대화 실행
                    conversation_log, final_dialogue = run_llm_conversation(
                        model_a,
                        model_b,
                        initial_prompt=prompt,
                        num_turns=num_turns
                    )

                    # 결과 저장
                    save_name = f"{problem_id}__{model_a_name}_x_{model_b_name}__turns{num_turns}.json"
                    save_path = os.path.join(RESULTS_DIR, save_name)

                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "problem_id": problem_id,
                            "model_pair": f"{model_a_name}_x_{model_b_name}",
                            "num_turns": num_turns,
                            "initial_prompt": prompt,
                            "final_dialogue": final_dialogue,
                            "conversation_log": conversation_log
                        }, f, indent=2, ensure_ascii=False)

                    print(f"Saved result: {save_path}")


if __name__ == "__main__":
    experiment()
