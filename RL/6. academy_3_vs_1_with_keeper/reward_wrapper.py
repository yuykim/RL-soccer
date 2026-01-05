import gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.prev_ball_x = None

    def step(self, action):
        # 환경의 기본 정보 (obs, reward, done, info)를 받아옴
        obs, reward, done, info = self.env.step(action)
        
        # simple115 관측값: index 88=공의 x좌표 (-1.0 ~ 1.0)
        ball_x = obs[88]

        # [커스텀 보상 로직]
        custom_reward = reward # 기본 골 보상(1.0) 포함

        # 공이 상대 골대 방향으로 전진할 때 보상 부여
        if self.prev_ball_x is not None:
            if ball_x > self.prev_ball_x:
                custom_reward += 0.01
        
        self.prev_ball_x = ball_x

        return obs, custom_reward, done, info

    def reset(self):
        self.prev_ball_x = None
        return self.env.reset()