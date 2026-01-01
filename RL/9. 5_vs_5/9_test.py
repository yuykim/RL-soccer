import os
import time
import gfootball.env as football_env
from stable_baselines3 import PPO
from utils import cleanup, save_frame, make_video, make_video_from_frames  # 영상 생성 유틸

SCENARIO_NAME = '5_vs_5'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "5_vs_5.zip")

if not os.path.exists(MODEL_PATH):
    print(f"error: can't find ({MODEL_PATH})")
    raise SystemExit(1)

# 테스트 환경 생성 (render=True로 눈으로 확인)
test_env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    stacked=False,
    representation='simple115',
    render=True
)

print(f"load model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

episodes = 1
for ep in range(1, episodes + 1):
    try:
        cleanup()
        obs = test_env.reset()
        done = False
        score = 0.0
        step_count = 0
        print(f"episode {ep} start!")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_out = test_env.step(action)

            # 프레임 저장
            frame = test_env.render(mode='rgb_array')
            save_frame(frame, step_count)

            # 종료 이유 출력 분기 처리
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
                if done:
                    print(f"Terminated(goal/out): {terminated}, Truncated(timeover): {truncated}")
            else:
                obs, reward, done, info = step_out
                if done:
                    print(f"Terminated(info): {info}")

            score += float(reward)
            step_count += 1
            time.sleep(0.05)

        print(f"episode {ep} terminated. score: {score}")
        time.sleep(1)
        make_video()

    except KeyboardInterrupt:
        # Ctrl+C 눌렸을 때: 지금까지 저장된 프레임으로 영상 생성
        print("\nCtrl+C detected! Creating emergency video from saved frames...")
        make_video()  # 이미 구현된 영상 생성 함수 호출

        # (선택) 프레임 폴더에서 직접 영상 만들기 함수가 따로 있다면:
        # make_video_from_frames("frames_saved_folder", "output_video.mp4")

        print("Video created. Exiting safely...")
        break  # 에피소드 루프 탈출

# 최종 종료 처리
try:
    test_env.close()
except:
    pass

print("finish!") 
