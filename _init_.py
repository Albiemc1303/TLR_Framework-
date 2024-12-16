# Initialize and train
if __name__ == "__main__":
    envs = ["CartPole-v1", "LunarLander-v2", "SpaceInvaders-ram-v0"]
    dqn_types = ["standard", "double", "dueling"]  # DQN variants for each environment
    manager = MultiEnvManager(envs, dqn_types, prioritized=True)
    manager.train(episodes=25000)
