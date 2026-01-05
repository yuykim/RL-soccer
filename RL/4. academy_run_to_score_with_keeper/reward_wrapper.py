import gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.prev_ball_x = None

    def step(self, action):
        # 환경의 기본 정보 받아오기
        obs, reward, done, info = self.env.step(action)
        
        # simple115 관측값: index 88=공의 x좌표, 89=공의 y좌표
        ball_x = obs[88]
        ball_y = obs[89]

        # [커스텀 보상 초기화]
        custom_reward = reward # 골 넣으면 받는 기본 보상(1.0) 포함

        # 1. 전진 보상 (기존 로직 유지)
        if self.prev_ball_x is not None:
            if ball_x > self.prev_ball_x:
                custom_reward += 0.01

        # 2. 골대 근처 접근 보상 (Proximity Reward)
        # 골대 정중앙(1.0, 0.0)에 가까워질수록 보상
        # x좌표가 0.7 이상(페널티 박스 근처)일 때 추가 보상
        if ball_x > 0.7:
            # 골대와의 거리를 계산하여 가까울수록 보상 (최대 0.05)
            distance_to_goal = np.sqrt((1.0 - ball_x)**2 + (0.0 - ball_y)**2)
            custom_reward += (0.05 * (1.0 - distance_to_goal))

        # 3. 슛 액션 유도 보상 (Shot Action Reward)
        # GRF에서 슛 액션의 인덱스는 12입니다.
        # 조건: 공이 상대 진영 깊숙이(x > 0.6) 있을 때 슛을 하면 보상 부여
        if action == 12: # 12 = Shot Action
            if ball_x > 0.6:
                custom_reward += 0.1  # 슛 시도 자체에 대한 격려
            else:
                custom_reward -= 0.05 # 너무 먼 거리에서 슛을 쏘는 '뻥축구' 방지

        self.prev_ball_x = ball_x

        return obs, custom_reward, done, info

    def reset(self):
        self.prev_ball_x = None
        return self.env.reset()