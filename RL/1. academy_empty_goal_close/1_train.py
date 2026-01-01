import os
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


# 1. 환경 및 학습 설정
SCENARIO_NAME = 'academy_empty_goal_close'              # 시나리오: 빈 골대에 골 넣기
TOTAL_TIMESTEPS = 50000                                 # 총 학습 단계 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")            # 모델 저장 폴더명
MODEL_NAME = "academy_empty_goal_close"                 # 저장될 모델 파일 이름
final_path = os.path.join(MODEL_DIR, MODEL_NAME)

# 저장 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. 학습용 환경 생성
# 중요: 학습 시에는 render=False로 설정하여 속도를 높입니다.
# representation='simple115': 복잡한 화면 대신 115개의 숫자로 요약된 정보를 사용합니다 (SB3 학습에 유리).
env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    stacked=False,
    representation='simple115',
    render=False
)

# 일정 간격마다 자동 저장 콜백 (체크포인트)
# save_freq는 "환경 step" 단위. 너무 잦으면 파일이 많이 생김.
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # 10000 step마다 저장 
    save_path=MODEL_DIR,
    name_prefix=f"{MODEL_NAME}_ckpt"
)


# print(f"[{SCENARIO_NAME}] 학습을 시작합니다...")
# print(f"관찰 공간(Observation Space): {env.observation_space}")
# print(f"행동 공간(Action Space): {env.action_space}")


# 3. 모델 생성 (PPO 알고리즘 사용)
# MlpPolicy: 입력이 이미지(픽셀)가 아닌 숫자 벡터('simple115')일 때 사용하는 신경망 구조
model = PPO("MlpPolicy", env, verbose=1)


# 4. 학습 (중간 저장 포함)
try:
    # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback) # 중간에 모델 저장 
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Ctrl+C로 끊어도 여기로 와서 저장하게 만들기
except KeyboardInterrupt:
    print("\ninterrupted")

finally:
    # 최종 모델 저장
    final_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(final_path)
    print(f"save complete: '{final_path}.zip'")

    env.close()