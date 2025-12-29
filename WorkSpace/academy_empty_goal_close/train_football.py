import os
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO

# ==========================================
# 1. 환경 및 학습 설정
# ==========================================
SCENARIO_NAME = 'academy_empty_goal_close' # 시나리오: 빈 골대에 골 넣기
TOTAL_TIMESTEPS = 50000                    # 총 학습 단계 (이 정도면 충분히 배웁니다)
MODEL_DIR = "models"                       # 모델 저장 폴더명
MODEL_NAME = "ppo_football_empty_goal"     # 저장될 모델 파일 이름

# 저장 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. 학습용 환경 생성
# ==========================================
# 중요: 학습 시에는 render=False로 설정하여 속도를 높입니다.
# representation='simple115': 복잡한 화면 대신 115개의 숫자로 요약된 정보를 사용합니다 (SB3 학습에 유리).
env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    stacked=False,
    representation='simple115',
    render=False
)

print(f"[{SCENARIO_NAME}] 학습을 시작합니다...")
print(f"관찰 공간(Observation Space): {env.observation_space}")
print(f"행동 공간(Action Space): {env.action_space}")

# ==========================================
# 3. 모델 생성 (PPO 알고리즘 사용)
# ==========================================
# MlpPolicy: 입력이 이미지(픽셀)가 아닌 숫자 벡터('simple115')일 때 사용하는 신경망 구조
model = PPO("MlpPolicy", env, verbose=1)

# ==========================================
# 4. 학습 시작!
# ==========================================
# 지정한 횟수만큼 환경과 상호작용하며 학습합니다.
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# ==========================================
# 5. 학습된 모델 저장
# ==========================================
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
model.save(model_path)
print(f"학습 완료! 모델이 '{model_path}.zip'에 저장되었습니다.")

env.close()