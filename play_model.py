import gym
import pygame
import time
import random

import torch

from model import *

if __name__ == '__main__':
    pygame.init()
    env=gym.make('CartPole-v1',render_mode='human')
    state,_=env.reset()

    cart_position=state[0]
    cart_speed=state[1]
    pole_position=state[2]
    pole_speed=state[3]

    model=Net()
    model.load_state_dict(torch.load('./cartpole.pth'))
    model.eval()

    print(f"begin state:{state}")
    print(f"cart position: {cart_position}")
    print(f"cart speed: {cart_speed}")
    print(f"pole position: {pole_position}")
    print(f"pole speed: {pole_speed}")
    time.sleep(3)

    start_time=time.time()
    max_action=1000

    step=0
    fail=False
    for step in range(max_action):
        print(step)
        time.sleep(0.3)

        state=torch.from_numpy(state).float().unsqueeze(0)
        probs=model(state)
        action=torch.argmax(probs,dim=1).item()

        state,_,done,_,_=env.step(action)
        if done:
            fail=True
            break

    end_time=time.time()
    game_time=end_time-start_time
    if fail:
        print(f"fail,you play {game_time:.2f} seconds, {step} steps")
    else:
        print(f"success,you play {game_time:.2f} seconds, {step} steps")
    env.close()