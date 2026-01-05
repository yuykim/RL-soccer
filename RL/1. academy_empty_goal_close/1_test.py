import os
import time
import gfootball.env as football_env
from stable_baselines3 import PPO
from utils import cleanup, save_frame, make_video # 직접 만드신 유틸리티 함수 가정

# 1. 시나리오 및 경로 설정
SCENARIO_NAME = 'academy_empty_goal_close' # 현재 테스트할 시나리오 이름

# 파일 위치 기준으로 상위 폴더의 models 폴더에서 모델을 찾습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, f"{SCENARIO_NAME}.zip")

# 모델 파일 존재 여부 확인
if not os.path.exists(MODEL_PATH):
    print(f"Error: Can't find model file at ({MODEL_PATH})")
    raise SystemExit(1)

# 2. 테스트용 환경 생성 
# 화면을 보기 위해 render=True, 학습 때와 동일한 simple115 사용
test_env = football_env.create_environment( 
    env_name=SCENARIO_NAME, 
    stacked=False, 
    representation='simple115', 
    render=True 
)

print(f">>> Load model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

# 3. 에피소드 반복 테스트
episodes = 5
for ep in range(1, episodes + 1):
    # cleanup() # 이전 프레임 삭제 등 정리

    obs = test_env.reset()
    done = False
    score = 0.0
    step_count = 0
    print(f">>> Episode {ep} start!")

    while not done:
        # 모델로부터 행동 예측 (deterministic=True로 설정하여 최선의 행동 유도)
        action, _ = model.predict(obs, deterministic=True)
        step_out = test_env.step(action)

        # 현재 화면 렌더링 및 프레임 저장
        # frame = test_env.render(mode='rgb_array')
        # save_frame(frame, step_count)

        # 환경 버전에 따른 반환값 개수 처리 (4개 또는 5개)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        score += float(reward)
        step_count += 1
        time.sleep(0.01) # VNC에서 너무 빠르지 않게 조절

    print(f">>> Episode {ep} terminated. Total score: {score}")
    time.sleep(1)

    # 한 에피소드가 끝나면 영상 제작
    print(f">>> Making video for episode {ep}...")
    # make_video()

test_env.close()
print(">>> All tests and video recordings finished!")