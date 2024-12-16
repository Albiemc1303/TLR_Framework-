# Multi-Environment Training Manager with Shared Buffer
class MultiEnvManager:
    def __init__(self, envs, dqn_types, prioritized=False):
        self.envs = [gym.make(env) for env in envs]
        self.shared_buffer = SharedReplayBuffer(BUFFER_SIZE) if prioritized else None
        self.agents = [
            DQNAgent(
                env.observation_space.shape[0],
                env.action_space.n,
                dueling=(dqn_type == "dueling"),
                double=(dqn_type == "double"),
                prioritized=prioritized,
                shared_buffer=self.shared_buffer
            ) for env, dqn_type in zip(self.envs, dqn_types)
        ]
        self.episode_logs = []

    def train(self, episodes, log_file="training_log.csv"):
        for episode in range(episodes):
            total_rewards = [0] * len(self.envs)
            states = [env.reset() for env in self.envs]
            dones = [False] * len(self.envs)

            while not all(dones):
                for i, (env, agent) in enumerate(zip(self.envs, self.agents)):
                    if not dones[i]:
                        action = agent.select_action(states[i])
                        next_state, reward, done, _ = env.step(action)
                        agent.buffer.add((states[i], action, reward, next_state, done))
                        if self.shared_buffer:
                            self.shared_buffer.add((states[i], action, reward, next_state, done))
                        agent.train()
                        total_rewards[i] += reward
                        states[i] = next_state
                        dones[i] = done

            for agent in self.agents:
                agent.update_target_network()

            # Logging
            avg_rewards = [total / episodes for total in total_rewards]
            self.episode_logs.append([episode] + total_rewards)

            # Epsilon decay
            for agent in self.agents:
                agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

            # Terminal Output
            print(f"Episode {episode + 1}/{episodes} - Rewards: {total_rewards}")

        # Save logs to CSV
        log_df = pd.DataFrame(self.episode_logs, columns=["Episode"] + [f"Env_{i+1}_Reward" for i in range(len(self.envs))])
        log_df.to_csv(log_file, index=False)
        print(f"Training log saved to {log_file}")
