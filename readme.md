# ⚽ AI 축구 강화학습 캠프: 환경 설치 가이드 (Windows)

본 가이드는 **Google Research Football** 환경을 구축하고, 강화학습 AI를 만들기 위한 필수 도구들을 설치하는 과정을 담고 있습니다.

---

## 🛠️ 1. 사전 준비 (Pre-requisites)

설치를 시작하기 전, 아래 두 가지를 확인하세요.

1. **Miniconda 설치**: [Miniconda 홈페이지](https://docs.conda.io/en/latest/miniconda.html)에서 **Windows 64-bit** 버전을 다운로드하여 설치하세요.
* **중요**: 설치 중 `Add Miniconda3 to my PATH` 항목에 체크하면 편리합니다.


2. **터미널 열기**: 설치 완료 후, **'Anaconda Prompt'** 또는 **'PowerShell'**을 실행합니다.

---

## 💻 2. 단계별 설치 명령어 (Step-by-Step)

아래 명령어들을 순서대로 **한 줄씩** 복사해서 입력하세요.

### **1단계: 가상환경 생성 및 빌드 도구 설치**

```bash
# 가상환경 생성 (Python 3.8 권장)
conda create -n gfootball python=3.8 -y
conda activate gfootball

# 필수 빌드 도구 설치
conda install -c conda-forge cmake boost ninja -y

```

### **2단계: 패키지 관리자 및 라이브러리 설치**

```bash
# 도구 버전 고정 (충돌 방지)
python -m pip install "pip<24.1" "setuptools<66"

# [핵심] 호환되는 Pygame 버전 설치
pip install pygame==2.0.1

# 의존성 패키지 설치
pip install psutil wheel opencv-python six absl-py scipy

# Gym 및 알고리즘 라이브러리 설치
pip install gym==0.21.0 --no-deps
pip install stable-baselines3 shimmy

```

### **3단계: G-Football 엔진 설치**

이 과정은 컴퓨터 사양에 따라 **5~10분**이 걸릴 수 있습니다.

```bash
pip install gfootball

```

---

## ✅ 3. 설치 확인 (Test)

설치가 잘 되었는지 확인하기 위해 직접 축구 게임을 실행해 봅니다.

```bash
python -m gfootball.play_game --action_set=full

```

> **성공 시**: 게임 화면이 뜨면 키보드 방향키와 `A`, `S`, `D`, `F` 키로 선수를 조작할 수 있습니다.

---

## 🚨 4. 주요 오류 및 해결 방법 (Troubleshooting)

30명이 설치하다 보면 반드시 발생하는 오류들입니다. 당황하지 말고 아래 해결법을 따르세요.

### 오류 1: `RuntimeError: Dynamic linking causes SDL downgrade!`

* **원인**: Pygame과 G-Football이 사용하는 그래픽 라이브러리(SDL2) 버전이 충돌할 때 발생합니다.
* **해결법**: 터미널에 아래 명령어를 입력하여 파일을 강제로 교체하세요.
```powershell
copy "%CONDA_PREFIX%\Lib\site-packages\pygame\SDL2.dll" "%CONDA_PREFIX%\Lib\site-packages\gfootball_engine\SDL2.dll" /y

```
### 오류 2: `error: Microsoft Visual C++ 14.0 or greater is required.`

* **원인**: 엔진 컴파일에 필요한 C++ 컴파일러가 컴퓨터에 없는 경우입니다.
* **해결법**: [Visual Studio Build Tools](https://www.google.com/search?q=https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/)를 다운로드하여 실행한 후, **"C++를 사용한 데스크톱 개발"** 항목을 체크하고 설치하세요. (설치 후 컴퓨터 재부팅 권장)

### 오류 3: `ImportError: cannot import name 'six' from 'sklearn.utils.fixes'`

* **원인**: 일부 패키지에서 `six` 라이브러리를 찾지 못하는 경우입니다.
* **해결법**: `pip install six`를 입력하여 명시적으로 설치해 주세요.

### 오류 4: 게임 화면이 너무 작거나 끊겨요

* **원인**: 노트북의 저사양 모드 혹은 해상도 설정 문제입니다.
* **해결법**: `env_name` 설정에서 `render=False`로 두고 학습시킨 뒤, 결과 확인할 때만 `render=True`를 사용하세요.

---
