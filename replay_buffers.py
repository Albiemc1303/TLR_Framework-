# Modified Replay Buffer with Prioritization
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha

    def add(self, transition, td_error=1.0):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(transition)
        self.priorities.append(max_priority if td_error == 0 else td_error)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], []

        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self.buffer[idx] for idx in indices]
        return transitions, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Small value to prevent zero priority

    def __len__(self):
        return len(self.buffer)

# Shared Replay Buffer
class SharedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        return zip(*transitions)
