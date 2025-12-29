import os
import gfootball.env as football_env
from stable_baselines3 import PPO
import time

SCENARIO_NAME = 'academy_pass_and_shoot_with_keeper'

# ✅ 현재 파이썬 파일(test_pass_score.py) 위치 기준으로 경로 만들기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_custom_pass_then_shoot_touchout_penalty.zip")

print("현재 작업 폴더(cwd):", os.getcwd())
print("모델 경로:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"오류: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
    exit()

test_env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    stacked=False,
    representation='simple115',
    render=True
)

model = PPO.load(MODEL_PATH)

for episode in range(1, 6):
    obs = test_env.reset()
    done = False
    score = 0
    print(f"패스 미션 {episode}번 시작!")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        score += reward

    print(f"미션 종료. 결과: {'성공(골!)' if score > 0 else '실패'}")
    time.sleep(1)

test_env.close()
