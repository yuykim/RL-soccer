import os
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
# 만들어둔 보상 함수 파일에서 클래스 가져오기
from reward_wrapper import CustomRewardWrapper 

# 1. 환경 및 학습 설정
SCENARIO_NAME = '5_vs_5'           # 현재 시나리오 이름
PREV_MODEL_NAME = '1_vs_1_easy'    # 이전 단계 모델 이름 (없으면 None)
TOTAL_TIMESTEPS = 50000 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. 학습용 환경 생성
env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    stacked=False,
    representation='simple115',
    render=False
)

# [핵심 추가] 직접 만든 커스텀 보상 함수로 환경 감싸기
env = CustomRewardWrapper(env)

# 3. 모델 생성 또는 로드 (전이 학습 로직)
prev_path = os.path.join(MODEL_DIR, f"{PREV_MODEL_NAME}.zip")
final_path = os.path.join(MODEL_DIR, SCENARIO_NAME)

if PREV_MODEL_NAME and os.path.exists(prev_path):
    print(f">>> Loading previous model: {prev_path}")
    model = PPO.load(prev_path, env=env)
else:
    print(f">>> No previous model found. Starting from scratch.")
    model = PPO("MlpPolicy", env, verbose=1)

# 체크포인트 설정
checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path=MODEL_DIR,
    name_prefix=f"{SCENARIO_NAME}_ckpt"
)

# 4. 학습 진행
try:
    print(f">>> Training started for {SCENARIO_NAME} with Custom Reward...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
except KeyboardInterrupt:
    print("\n>>> Interrupted by user")
finally:
    model.save(final_path)
    print(f">>> Save complete: '{final_path}.zip'")
    env.close()