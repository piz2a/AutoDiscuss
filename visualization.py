import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# experiment_results 디렉토리
RESULTS_DIR = "experiment_results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# GPT-4.1 기준 요금
PRICE_PER_INPUT_TOKEN = 2.00 / 1_000_000  # $2.00 per 1M input tokens
PRICE_PER_OUTPUT_TOKEN = 8.00 / 1_000_000  # $8.00 per 1M output tokens

# 결과 파일 읽기
def load_results():
    data = []
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]

    for file_name in tqdm(result_files, desc="Loading results"):
        file_path = os.path.join(RESULTS_DIR, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        # conversation_log에서 토큰과 시간 합산
        input_tokens = sum(turn.get("input_tokens", 0) for turn in result["conversation_log"])
        output_tokens = sum(turn.get("output_tokens", 0) for turn in result["conversation_log"])
        total_tokens = input_tokens + output_tokens
        elapsed_time = sum(turn.get("duration", 0) for turn in result["conversation_log"])

        # 비용 계산
        cost = (
            input_tokens * PRICE_PER_INPUT_TOKEN +
            output_tokens * PRICE_PER_OUTPUT_TOKEN
        )

        # 결과 저장
        data.append({
            "problem_id": result["problem_id"],
            "domain": result["problem_id"].split("_")[0],
            "model_pair": result["model_pair"],
            "num_turns": result["num_turns"],
            "score": result["grading_result"]["score"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "cost_usd": cost
        })

    df = pd.DataFrame(data)
    return df

# 시각화 함수
def plot_all_results(df):
    # Experiment 1만 필터링 (GPT-GPT)
    df_exp1 = df[df["model_pair"] == "gpt_x_gpt"].copy()
    # x축 순서 명시 (1, 2, 4, 8)
    df_exp1["num_turns_str"] = pd.Categorical(
        df_exp1["num_turns"].astype(str),
        categories=["1", "2", "4", "8"],
        ordered=True
    )
    # hue 색상 및 순서 통일 (math, writing, ps)
    domain_order = ["math", "writing", "ps"]
    df_exp1["domain"] = pd.Categorical(
        df_exp1["domain"],
        categories=domain_order,
        ordered=True
    )

    # 스타일
    sns.set(style="whitegrid")

    ### Plot 1: Score vs Num Turns (boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_exp1, x="num_turns_str", y="score", hue="domain")
    plt.title("Score vs Num Turns")
    plt.xlabel("Num Turns")
    plt.ylabel("Score (Accuracy)")
    plt.legend(title="Domain")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "score_vs_num_turns.png"))
    plt.close()

    ### Plot 2: Cost vs Num Turns (boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_exp1, x="num_turns_str", y="cost_usd", hue="domain")
    plt.title("Cost (USD) vs Num Turns")
    plt.xlabel("Num Turns")
    plt.ylabel("Cost (USD)")
    plt.legend(title="Domain")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cost_vs_num_turns.png"))
    plt.close()

    ### Plot 3: Elapsed Time vs Num Turns (boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_exp1, x="num_turns_str", y="elapsed_time", hue="domain")
    plt.title("Elapsed Time vs Num Turns")
    plt.xlabel("Num Turns")
    plt.ylabel("Elapsed Time (sec)")
    plt.legend(title="Domain")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elapsed_time_vs_num_turns.png"))
    plt.close()

    print(f"Plots saved to: {PLOTS_DIR}")


# 시각화 함수
def plot_model_comparison(df):
    # 실험 2 조건: num_turns == 4
    df_exp2 = df[df["num_turns"] == 4]

    # 스타일
    sns.set(style="whitegrid")

    # 모델 순서 고정 (원하는 순서로 보기 좋게 나열)
    model_order = ["gpt_x_gpt", "gpt_x_deepseek", "deepseek_x_gpt", "deepseek_x_deepseek"]
    df_exp2["model_pair"] = pd.Categorical(df_exp2["model_pair"], categories=model_order, ordered=True)
    df_exp2 = df_exp2.sort_values("model_pair")

    # domain 순서 통일 (math, writing, ps)
    domain_order = ["math", "writing", "ps"]
    df_exp2["domain"] = pd.Categorical(df_exp2["domain"], categories=domain_order, ordered=True)

    ### Plot 1: Score vs Model Pair (domain별 swarmplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_exp2, x="model_pair", y="score", hue="domain", dodge=True)
    plt.title("Score vs Model Pair by Domain")
    plt.xlabel("Model Pair")
    plt.ylabel("Score (Accuracy)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "score_vs_model_pair.png"))
    plt.close()

    ### Plot 2: Cost vs Model Pair (domain별 swarmplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_exp2, x="model_pair", y="cost_usd", hue="domain", dodge=True)
    plt.title("Cost (USD) vs Model Pair by Domain")
    plt.xlabel("Model Pair")
    plt.ylabel("Cost (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cost_vs_model_pair.png"))
    plt.close()

    ### Plot 3: Elapsed Time vs Model Pair (domain별 swarmplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_exp2, x="model_pair", y="elapsed_time", hue="domain", dodge=True)
    plt.title("Elapsed Time vs Model Pair by Domain")
    plt.xlabel("Model Pair")
    plt.ylabel("Elapsed Time (sec)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elapsed_time_vs_model_pair.png"))
    plt.close()

    print(f"Plots saved to: {PLOTS_DIR}")


# 실행 예시
if __name__ == "__main__":
    df = load_results()
    plot_all_results(df)
    plot_model_comparison(df)
