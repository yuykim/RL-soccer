import os
import time
import gfootball.env as football_env
from stable_baselines3 import PPO
from utils import cleanup, save_frame, make_video

SCENARIO_NAME = 'academy_empty_goal_close'

# 파일 위치 기준으로 모델 경로 고정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "academy_empty_goal_close.zip")

if not os.path.exists(MODEL_PATH):
    print(f"error: can't find ({MODEL_PATH})")
    raise SystemExit(1)

# 2. 테스트용 환경 생성 
# 중요: 이번에는 눈으로 봐야 하므로 render=True로 설정합니다. 
# representation은 반드시 학습할 때와 같아야 합니다 ('simple115'). 
test_env = football_env.create_environment( 
    env_name=SCENARIO_NAME, 
    stacked=False, 
    representation='simple115', 
    render=True 
    )

print(f"load model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

episodes = 5
for ep in range(1, episodes + 1):
    cleanup()

    obs = test_env.reset()
    done = False
    score = 0.0
    step_count = 0
    print(f"episode {ep} start!")

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        step_out = test_env.step(action)

        frame = test_env.render(mode='rgb_array')
        save_frame(frame, step_count)

        # 4개/5개 리턴 모두 대응
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        score += float(reward)
        step_count += 1
        time.sleep(0.05)

    print(f"episode {ep} terminated. score: {score}")
    time.sleep(1)

    make_video()

test_env.close()
print("finish!")
