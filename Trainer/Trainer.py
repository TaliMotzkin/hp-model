

class Trainer:
    def __init__(
        self,
        env,
        agent,
        cfg= None):
        self.env = env
        self.agent = agent


def train_dqn(self):
    states, infos = self.env.reset()

    all_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        all_rewards.append(total_reward)

        # Decay epsilon
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        # Update target network every X episodes
        if episode % target_update == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    return all_rewards
