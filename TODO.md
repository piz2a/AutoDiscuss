# To-Do: AutoDiscuss - LLM-to-LLM Experiment Framework

---

## ✅ 목표

- LLM끼리 대화하여 문제 해결 성능 분석
- 정확도, 비용, 시간 측정 및 비교
- GPT / DeepSeek 모델별 비교 분석

---

## 📌 현재 진행률 (Progress)

**진행률**: 약 **60%**  
**구조 설계**는 거의 완료, **주요 구현(채점/통계)** 단계 진입

---

## 🗂️ 할 일 (To-Do List)

### 1️⃣ grade_ps_problem 완성 및 디버깅 ✅

- [ ] 함수 안정화 (파싱 + 실행 + 채점 확인)
- [ ] 실제 PS 문제 테스트 케이스 적용

---

### 2️⃣ grade_exam_problem 구현 📝

- [X] GPT로 재채점 구조 설계
- [X] criteria 기반 점수 계산
- [X] 테스트 Exam 문제 적용

---

### 3️⃣ experiment() 수정 → 채점 통합

- [X] PS 문제는 grade_ps_problem 사용
- [X] Exam 문제는 grade_exam_problem 사용
- [X] 대화 결론 파싱 + 채점 결과 저장

---

### 4️⃣ 정확도 / 비용 / 시간 측정 기능 추가

- [X] 정확도 저장
- [ ] 비용 계산 (tokens * API 요금)
- [ ] 시간 측정 (start ~ end)

---

### 5️⃣ 결과 시각화

- [ ] 표 출력
- [ ] 그래프 출력 (모델/문제 분야/turn 별)
- [ ] 프로토타입 웹서비스 개발 (발표 도중 모두가 들어가서 사용해볼 수 있게)
