import os
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO
import time

# ==========================================
# 1. 설정 (학습 때와 맞춰야 함)
# ==========================================
SCENARIO_NAME = 'academy_empty_goal_close'
MODEL_PATH = "models/ppo_football_empty_goal.zip" # 저장된 모델 경로

# 모델 파일이 있는지 확인
if not os.path.exists(MODEL_PATH):
    print(f"오류: 모델 파일({MODEL_PATH})을 찾을 수 없습니다. 먼저 train_football.py를 실행하세요.")
    exit()

# ==========================================
# 2. 테스트용 환경 생성
# ==========================================
# 중요: 이번에는 눈으로 봐야 하므로 render=True로 설정합니다.
# representation은 반드시 학습할 때와 같아야 합니다 ('simple115').
test_env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    stacked=False,
    representation='simple115',
    render=True
)

# ==========================================
# 3. 저장된 모델 불러오기
# ==========================================
print(f"모델을 불러옵니다: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

# ==========================================
# 4. 게임 실행 루프
# ==========================================
Episodes = 5 # 테스트할 게임 횟수

for episode in range(1, Episodes + 1):
    obs = test_env.reset()
    done = False
    score = 0
    
    print(f"에피소드 {episode} 시작!")
    
    while not done:
        # 학습된 모델에게 현재 상황(obs)을 보여주고, 최적의 행동(action)을 예측하게 함
        # deterministic=True: 학습된 대로만 가장 확률 높은 행동을 하라 (탐험 X)
        action, _states = model.predict(obs, deterministic=True)
        
        # 환경에 행동 적용
        obs, reward, done, info = test_env.step(action)
        score += reward
        
        # 너무 빠르면 보기 힘들므로 약간의 지연 추가 (선택 사항)
        # time.sleep(0.05)

    print(f"에피소드 {episode} 종료. 획득 점수: {score}")
    time.sleep(1) # 다음 게임 시작 전 잠시 대기

test_env.close()
print("테스트 종료!")