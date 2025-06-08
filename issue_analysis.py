import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from llmapi_integration import LLMAPI
from problems import load_math_problems, load_ps_problems, load_writing_problems, grade_math_problem, grade_writing_problem, grade_ps_problem

# 세팅
RESULTS_DIR = "experiment_results"
SLICE_COUNTS = [1, 2, 3, 4]

# API 키 로드
with open('api_key.json', 'r', encoding='utf-8') as f:
    api_keys = json.load(f)

grader = LLMAPI(model='gpt', api_key=api_keys.get('gpt'))

# 문제 로드 (채점용)
problems_dict = {
    "math": load_math_problems("problems/math.json"),
    "writing": load_writing_problems("problems/writing.json"),
    "ps": load_ps_problems("problems/ps.json"),
}

def get_results():
    # 결과 저장용 DataFrame
    results = []

    # JSON 파일 필터링 (GPT x GPT, 8턴)
    target_files = [
        fname for fname in os.listdir(RESULTS_DIR)
        if fname.endswith(".json") and "gpt_x_gpt" in fname and "turns4" in fname and "math" in fname
    ]

    print(f"Found {len(target_files)} target files.")

    # 슬라이싱 후 채점
    for fname in tqdm(target_files):
        filepath = os.path.join(RESULTS_DIR, fname)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        domain = data['problem_id'].split("_")[0]
        problem_id = data['problem_id']
        conversation_log = data['conversation_log']
        problem_obj = None

        # 문제 매칭
        idx = int(problem_id.split("_q")[-1]) - 1
        problem_obj = problems_dict[domain][idx]

        for slice_count in SLICE_COUNTS:
            # 슬라이싱 적용
            sliced_log = conversation_log[:slice_count]
            if not sliced_log:
                continue

            final_answer_text = sliced_log[-1]["response_text"].strip()

            # 채점
            if domain == "math":
                score, grader_reply = grade_math_problem(problem_obj, final_answer_text, grader)
            elif domain == "writing":
                score, grader_reply = grade_writing_problem(problem_obj, final_answer_text, grader)
            elif domain == "ps":
                score, grader_reply = grade_ps_problem(problem_obj, final_answer_text)
            else:
                continue  # 잘못된 도메인 skip

            # 결과 저장
            results.append({
                "problem_id": problem_id,
                "domain": domain,
                "slice_count": slice_count,
                "score": score
            })
            print(results[-1])

    # Save results to JSON
    if not os.path.exists('issues'):
        os.makedirs('issues')

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'issues/issue_analysis_{timestamp}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_filename}")

    return results


results_files = [
    fname for fname in os.listdir('issues')
    if fname.endswith(".json") and "issue_analysis" in fname
]
print(results_files)
if not results_files:
    issue_results = get_results()
else:
    with open(os.path.join('issues', results_files[0]), 'r') as f:
        issue_results = json.load(f)

# DataFrame 생성
df = pd.DataFrame(issue_results)
df["slice_count"] = pd.Categorical(df["slice_count"], categories=[1, 2, 3, 4], ordered=True)

# 박스플롯
plt.figure(figsize=(6, 6))
sns.lineplot(data=df, x="slice_count", y="score", hue="problem_id", marker="o", legend=False)
plt.xticks([1, 2, 3, 4])
plt.title("Score vs Slice Count Analysis (GPTxGPT, 4 turns)")
plt.xlabel("Slice Count (Turns used)")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("plots/issue_analysis_boxplot.png")
plt.show()
