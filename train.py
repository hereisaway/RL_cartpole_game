import torch
import gym
from model import *
from torch.distributions import Categorical


def compute_loss(n,p):
    r=list()
    for i in range(n,0,-1):
        r.append(i*1.0)
    r=torch.tensor(r)

    r=(r-r.mean ())/r.std()
    loss=0
    for pi,ri in zip(p,r):
        loss += -pi*ri
    return loss

if __name__=='__main__':
    env=gym.make('CartPole-v1')
    env.reset(seed=543)
    torch.manual_seed(543)

    model=Net()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

    max_episode=1000#最大回合数
    max_action=10000#每回合最大行动数
    max_step=5000#完成训练的步数

    for ep in range(max_episode):
        state,_=env.reset()
        step=0
        p=list()
        for step in range(1,max_action+1):
            state=torch.from_numpy(state).float().unsqueeze(0)
            probs=model(state)

            m=Categorical(probs)
            action=m.sample()

            state,_,done,_,_=env.step(action.item())
            if done:
                break
            p.append(m.log_prob(action))

        if(step>max_step):
            print(f"training done!last episode: {ep},last step: {step}")
            break

        optimizer.zero_grad()
        loss=compute_loss(step,p)
        loss.backward()
        optimizer.step()
        if ep%10==0:
            print(f'Episode: {ep}, step: {step}, loss: {loss}')

    torch.save(model.state_dict(),'./cartpole.pth')
    env.close()

