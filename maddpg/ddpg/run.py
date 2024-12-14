import numpy as np
from agent import Agent
from env import MultiUAVEnv

def run():
    env = MultiUAVEnv()
    obs = env.reset()
    uav_ids = sorted(env.uavs.keys())  # 获取所有无人机的ID
    n_agents = len(uav_ids)

    agents = []
    for uav_id in uav_ids:
        input_dim = env.observation_spaces[uav_id].shape[0]
        n_action = env.action_spaces[uav_id].shape[0]
        agents.append(Agent(input_dims=input_dim, n_actions=n_action,
                            gamma=0.95, tau=0.01, alpha=1e-4, beta=1e-3))

    EVAL_INTERVAL = 50
    MAX_STEPS = 5000

    total_steps = 0
    episode = 0

    eval_scores = []
    eval_steps = []
    score = evaluate(agents, env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    while total_steps < MAX_STEPS:
        obs = env.reset()
        terminal = [False] * n_agents
        obs = [obs[uav_id] for uav_id in uav_ids]  # 按无人机ID顺序排列观测
        while not any(terminal):
            action = [agent.choose_action(obs[idx]) for idx, agent in enumerate(agents)]
            action = {uav_id: act for uav_id, act in zip(uav_ids, action)}
            obs_, reward, done, truncated, info = env.step(action)

            list_done = [done[uav_id] for uav_id in uav_ids]
            list_trunc = [truncated[uav_id] for uav_id in uav_ids]
            list_reward = [reward[uav_id] for uav_id in uav_ids]
            list_action = list(action.values())
            obs_ = [obs_[uav_id] for uav_id in uav_ids]

            terminal = [d or t for d, t in zip(list_done, list_trunc)]

            for idx, agent in enumerate(agents):
                agent.remember(obs[idx], list_action[idx],
                               list_reward[idx], obs_[idx], terminal[idx])

            if total_steps % 100 == 0:
                for agent in agents:
                    # print(f"[DEBUG] Step {total_steps} - Learning with Replay Buffer Size: {agent.memory.mem_cntr}")
                    agent.learn()
            obs = obs_
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0 and total_steps > 0:
            score = evaluate(agents, env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        episode += 1

        np.save('../data/my_ddpg_scores.npy', np.array(eval_scores))
        np.save('../data/my_ddpg_steps.npy', np.array(eval_steps))

def evaluate(agents, env, ep, step):
    score_history = []
    for i in range(3):
        obs = env.reset()
        score = 0
        uav_ids = sorted(env.uavs.keys())
        terminal = [False] * len(uav_ids)
        obs = [obs[uav_id] for uav_id in uav_ids]
        while not any(terminal):
            action = [agent.choose_action(obs[idx], eval=True)
                      for idx, agent in enumerate(agents)]
            action = {uav_id: act for uav_id, act in zip(uav_ids, action)}

            obs_, reward, done, truncated, info = env.step(action)
            obs_ = [obs_[uav_id] for uav_id in uav_ids]
            list_done = [done[uav_id] for uav_id in uav_ids]
            list_trunc = [truncated[uav_id] for uav_id in uav_ids]
            list_reward = [reward[uav_id] for uav_id in uav_ids]

            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            obs = obs_
            score += sum(list_reward)
        score_history.append(score)
    avg_score = np.mean(score_history)
    print(f'Evaluation episode {ep} train steps {step} average score {avg_score:.1f}')

    return avg_score

if __name__ == '__main__':
    run()
