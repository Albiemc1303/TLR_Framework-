def reward_shaping(self, env_name, state, action, reward, next_state, done, step_count=0):
        # Reward shaping for CartPole-v1
        shaped_reward = reward
        
        if env_name == "CartPole-v1":
            pole_angle = abs(next_state[2])  # Angle of the pole
            cart_velocity = abs(next_state[1])  # Cart's velocity
            shaped_reward -= 0.1 * pole_angle  # Penalize larger angles
            shaped_reward -= 0.1 * cart_velocity  # Penalize fast movements
            if done and step_count >= 200: 
                shaped_reward += 10  # Success bonus for 200 steps
        elif env_name == "LunarLander-v2":
            x_position = abs(next_state[0])  # Horizontal distance
            fuel_penalty = abs(next_state[4])  # Simulated fuel cost
            shaped_reward -= 0.5 * fuel_penalty  # Penalize fuel usage
            if abs(x_position) < 0.1 and abs(next_state[1]) < 0.1:
                shaped_reward += 5.0  # Landing bonus
            if done and abs(next_state[3]) < 0.5:
                shaped_reward += 50  # Soft landing
        elif env_name == "SpaceInvaders-ram-v0":
            enemies_destroyed = reward // 10  # Use reward to infer destroyed enemies
            shaped_reward += 1.0 * enemies_destroyed  # Reward for each enemy destroyed
            if action == 1:
                shaped_reward += 2  # Reward for shooting
            if done:
                shaped_reward -= 5  # Penalize game-over state
        
        # Normalize the shaped reward
        shaped_reward = normalize_reward(shaped_reward)
        return shaped_reward

    def train(self, episodes, log_file="training_log.csv"):
        for episode in range(episodes):
            total_rewards = [0] * len(self.envs)
            states = [env.reset() for env in self.envs]
            dones = [False] * len(self.envs)

            while not all(dones):
             for i, (env, agent) in enumerate(zip(self.envs, self.agents)):
                if not dones[i]:
                    if isinstance(agent, PPOAgent):
                        action, log_prob = agent.select_action(states[i])
                        next_state, reward, done, _ = env.step(action)

                        # Apply Reward Shaping
                        shaped_reward = self.reward_shaping(env.spec.id, states[i], action, reward, next_state, done)

                        agent.store_transition((states[i], action, shaped_reward, log_prob, done))
                    else:
                        action = agent.select_action(states[i])
                        next_state, reward, done, _ = env.step(action)

                        # Apply Reward Shaping
                        shaped_reward = self.reward_shaping(env.spec.id, states[i], action, reward, next_state, done)

                        agent.buffer.add((states[i], action, shaped_reward, next_state, done))
                        agent.train()

                    total_rewards[i] += shaped_reward
                    states[i] = next_state
                    dones[i] = done


            for agent in self.agents:
                if isinstance(agent, PPOAgent):
                    agent.train()
                else:
                    agent.update_target_network()

def normalize_reward(reward, min_val=-1, max_val=1):
    """Normalizes rewards to a specific range."""
    return max(min(reward, max_val), min_val)

