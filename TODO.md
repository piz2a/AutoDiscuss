# To-Do: AutoDiscuss - LLM-to-LLM Experiment Framework

---

## ✅ 목표

- LLM끼리 대화하여 문제 해결 성능 분석
- 정확도, 비용, 시간 측정 및 비교
- 대화 횟수에 따른 성능 변화 분석
- GPT / DeepSeek 모델별 비교 분석
- 안정적이고 재현 가능한 실험 프레임워크 구축

---

## 📌 현재 진행률 (Progress)

**진행률**: 약 **80%**  
**구조 설계 및 주요 구현 완료**  
**통계/시각화 단계 진입 중**

---

## 🗂️ 할 일 (To-Do List)

### 🧩 핵심 기능

- [X] grade_math_problem 통합
- [X] grade_writing_problem 통합
- [X] grade_ps_problem 통합
- [X] Grader reply 저장 기능 구현
- [X] DeepSeek token 처리 대응 (prompt_tokens, completion_tokens)
- [X] experiment() → 대화횟수 비교를 최우선으로 변경

---

### 📊 정확도 / 비용 / 시간 측정

- [X] 정확도 저장
- [X] 비용 계산 (tokens * API 요금)
- [X] 시간 측정 (start ~ end)
- [X] 결과 JSON 통합 저장

---

### 📝 실험 품질 개선

- [X] system 프롬프트 수정:
    - 협력 강조 문구 제거
    - "너희끼리 대화하여 하나의 AI보다 더 나은 답변을 만들어야 한다"는 목표 명시
- [X] 수학 문제용 해설 JSON 구조 설계 (문제 + 해설 포함)
- [X] grading 단계에서 해설 자동 파싱 → 프롬프트 수정 (해설 + 평가 기준 함께 제공)
- [X] 결론에는 계산 등 모든 논리 과정을 표현하도록 프롬프트 수정
- [X] 수학은 채점 기준과 등가인 식이 존재하면 점수를 부여하는 방식으로 평가 기준을 완화하였음

---

### 📝 테스트/유닛 테스트 도입

- [ ] PyTest 구조 도입 (grade 함수 등 테스트)
- [ ] 실험 코드 검증 자동화

---

### 🕹️ 실험 UX 개선

- [X] experiment() pause/resume 기능 구현
- [X] 실험 진행률 표시 (진행률 % 출력)

---

### 📈 결과 시각화

- [ ] 표 출력 (모델/문제 분야/turn 별 성능)
- [ ] 그래프 출력 (모델/문제 분야/turn 별)
- [ ] 프로토타입 웹서비스 개발 (실험 결과 시각화)

---

## 🚀 향후 고도화 아이디어

- [ ] Grader 모델 다양화 (GPT-4o / Claude / DeepSeek 등 비교)
- [ ] Chat prompt engineering (domain-specific tuning)
- [ ] 실험 자동화 스크립트 (all experiments batch 실행)
