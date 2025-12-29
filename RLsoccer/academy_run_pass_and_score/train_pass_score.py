import os
import gym
import gfootball.env as football_env
from stable_baselines3 import PPO

# ==========================================
# 1. 커스텀 보상 Wrapper
#    - 패스 성공 후 일정 step 내 슛: 큰 보상
#    - 터치아웃(라인 아웃): 페널티
# ==========================================
class PassThenShootRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        shot_action_id: int = 12,
        window_steps: int = 30,
        out_penalty: float = -0.2,
        out_threshold: float = 1.0,
        pass_success_bonus: float = 0.03,
        shoot_bonus: float = 0.01,
        pass_then_shoot_bonus: float = 0.15,
        own_ball_bonus: float = 0.001,
        forward_bonus: float = 0.01,
    ):
        super().__init__(env)
        self.prev_raw = None

        # 상태 변수
        self.last_pass_step = None
        self.step_count = 0

        # 하이퍼파라미터
        self.shot_action_id = shot_action_id
        self.window_steps = window_steps

        # 보상 파라미터
        self.out_penalty = out_penalty
        self.out_threshold = out_threshold

        self.pass_success_bonus = pass_success_bonus
        self.shoot_bonus = shoot_bonus
        self.pass_then_shoot_bonus = pass_then_shoot_bonus
        self.own_ball_bonus = own_ball_bonus
        self.forward_bonus = forward_bonus

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # gfootball raw 관측 (dict)
        raw = self.env.unwrapped.observation()[0]

        # -----------------------------
        # (A) 기본 shaping
        # -----------------------------
        # 1) 공 소유(아군) 작은 보상
        if raw.get("ball_owned_team", -1) == 0:
            reward += self.own_ball_bonus

        # 2) 공 전진 보상 (아군 소유일 때 x가 증가하면 보상)
        ball_x, ball_y, ball_z = raw["ball"][0], raw["ball"][1], raw["ball"][2]
        if self.prev_raw is not None:
            prev_ball_x = self.prev_raw["ball"][0]
            if raw.get("ball_owned_team", -1) == 0 and ball_x > prev_ball_x:
                reward += self.forward_bonus

        # -----------------------------
        # (NEW) 터치아웃 페널티
        # -----------------------------
        # 필드 밖으로 나가면(대략 x,y 범위가 [-1,1]) 페널티
        out_of_bounds = (abs(ball_x) > self.out_threshold) or (abs(ball_y) > self.out_threshold)
        if out_of_bounds:
            # 기본 페널티
            reward += self.out_penalty
            # 아군이 공을 가지고 있다가 라인 아웃되면 더 큰 페널티 (2배)
            if raw.get("ball_owned_team", -1) == 0:
                reward += self.out_penalty

        # -----------------------------
        # (B) 패스 성공 감지
        # -----------------------------
        # 아군(0) 소유 상태에서 ball_owned_player가 바뀌면 "패스 성공"으로 간주
        if self.prev_raw is not None:
            prev_team = self.prev_raw.get("ball_owned_team", -1)
            prev_player = self.prev_raw.get("ball_owned_player", -1)

            cur_team = raw.get("ball_owned_team", -1)
            cur_player = raw.get("ball_owned_player", -1)

            if (
                prev_team == 0 and cur_team == 0
                and prev_player != -1 and cur_player != -1
                and prev_player != cur_player
            ):
                self.last_pass_step = self.step_count
                reward += self.pass_success_bonus

        # -----------------------------
        # (C) 슈팅 보상
        # -----------------------------
        if action == self.shot_action_id:
            # 그냥 슛 보상(약하게)
            reward += self.shoot_bonus

            # 패스 이후 window_steps 안에 슛이면 추가 보상(크게)
            if self.last_pass_step is not None:
                if (self.step_count - self.last_pass_step) <= self.window_steps:
                    reward += self.pass_then_shoot_bonus

        # bookkeeping
        self.prev_raw = raw
        self.step_count += 1

        # 에피소드 종료 시 상태 초기화
        if done:
            self.prev_raw = None
            self.last_pass_step = None
            self.step_count = 0

        return obs, reward, done, info

    def reset(self):
        self.prev_raw = None
        self.last_pass_step = None
        self.step_count = 0
        return self.env.reset()


# ==========================================
# 2. 환경 생성 및 학습
# ==========================================
SCENARIO_NAME = "academy_pass_and_shoot_with_keeper"
TOTAL_TIMESTEPS = 100000

# ✅ 저장 경로: "이 파일 위치 기준" models 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = "ppo_custom_pass_then_shoot_touchout_penalty"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# 기본 환경 생성
env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    representation="simple115",
    render=False
)

# ✅ 커스텀 보상 Wrapper 적용
env = PassThenShootRewardWrapper(
    env,
    shot_action_id=12,
    window_steps=30,
    out_penalty=-0.2,     # 터치아웃 페널티 (너무 세면 -0.1부터 시작 추천)
    out_threshold=1.0,    # 감지가 안 되면 1.02~1.05로 조정
    pass_success_bonus=0.03,
    shoot_bonus=0.01,
    pass_then_shoot_bonus=0.15,
    own_ball_bonus=0.001,
    forward_bonus=0.01,
)

print(f"[{SCENARIO_NAME}] '패스 후 슛 + 터치아웃 페널티' 커스텀 보상 학습을 시작합니다...")
print("모델 저장 경로:", MODEL_PATH + ".zip")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# 모델 저장
os.makedirs(MODELS_DIR, exist_ok=True)
model.save(MODEL_PATH)
print("학습 완료 및 저장 성공!")
