# ⚽ RL-Soccer: Imitation Learning (Behavior Cloning)

이 프로젝트는 **Google Research Football(G-Football)** 환경에서 인간 전문가의 플레이 데이터를 수집하고, 이를 바탕으로 인공지능이 축구 정책을 학습하는 **이미테이션 러닝(Imitation Learning)** 시스템을 구현합니다.

---
## 실제 플레이
![Robot Arm Football Demo](../doc/gameplay.gif)

- 사람의 게임 플레이영상 : https://drive.google.com/file/d/1nhdAwGLTyb9hywAcStkV3nlO3njdwcVK/view?usp=sharing
    - (약 6000프레임)
- 학습된 모델의 플레이 영상 : https://drive.google.com/file/d/1fYemqlV27fp6omK9S_TlL1f8KAjSiQB-/view?usp=sharing

---

## 📂 프로젝트 구조 (Project Structure)

```text
RL-soccer/
└── IL/
    ├── collect_data.py   # 전문가 플레이 데이터 수집 (게임패드/키보드 지원)
    ├── train_il.py       # Behavior Cloning 기반 신경망 학습
    ├── test_il.py        # 학습된 모델 성능 테스트 및 검증
    ├── expert_data.npz   # 수집된 상태(Obs)-액션(Action) 데이터셋
    └── il_model.pth      # 학습된 PyTorch 모델 가중치

```

---

## 주요 특징 (Key Features)

### 1. 하이브리드 입력 시스템 (Dual Input Support)

* **게임패드(Gamepad)** 및 **키보드(Keyboard)** 입력을 모두 지원하여 데이터 수집의 편의성을 극대화했습니다.
* 조작 방식에 상관없이 모든 입력은 `CoreAction` 객체로 변환되어 일관된 데이터 포맷으로 저장됩니다.

### 2. 정제된 전문가 의사결정 (Refined Expert Decision)

* 단순한 실시간 플레이 수집을 넘어, **팀 단위 토론**을 통해 결정된 최적의 액션을 데이터셋에 반영합니다.
* 이는 인간의 조작 실수를 배제하고, 모델이 **최적 정책()**에 더 빠르게 수렴하도록 돕습니다.

### 3. 동적 모델 아키텍처 (Dynamic Architecture)

* 데이터셋에 포함된 액션의 가짓수(최대 32개)를 학습 시 자동으로 파악하여 모델 구조를 생성합니다.
* 테스트 시 확률 기반 **소프트맥스 샘플링(Softmax Sampling)**을 적용하여 AI의 움직임이 고착화되는 것을 방지하고 역동적인 플레이를 구현했습니다.

---

## 🛠️ 사용 방법 (Usage)

### Step 1. 데이터 수집 (Data Collection)

적당한 속도(18 FPS)에서 전문가의 플레이를 기록합니다.

```bash
python IL/collect_data.py

```

### Step 2. 모델 학습 (Training)

수집된 `expert_data.npz`를 기반으로 신경망을 학습시킵니다.

```bash
python IL/train_il.py

```

* **Loss Function**: Cross Entropy Loss
* **Optimization**: Adam Optimizer

### Step 3. AI 테스트 (Testing)

학습된 모델이 실제 경기에서 어떻게 작동하는지 확인합니다.

```bash
python IL/test_il.py

```

---

## 📊 기술 데이터 규격 (Technical Specification)

### State Space (Observation)

* **Representation**: `simple115v2`
* **Dimension**: 115 (공의 위치/속도, 우리팀/상대팀 선수 22명의 좌표 및 속도 정보 포함)

### Action Space

* **Action Set**: `Full` (32 Actions)
* **Type**: Discrete (이산적 액션 제어)

---
