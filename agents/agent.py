# TODO: your agent here!
import numpy as np
from task import Task
import random
from collections import namedtuple, deque
from agents.actor import Actor
from agents.critic import Critic

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class MyAgent():
    def __init__(self, task):
        # My agent uses Deep Deterministic Policy Gradients (DDPG) to learn
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = task.action_high - task.action_low

        # Actor Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low ,self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low ,self.action_high)
        
        # Critic Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Noise Ornstein–Uhlenbeck
        self.exploration_mu = 0
        self.exploration_theta =  0.1 #0.15 , 0.1
        self.exploration_sigma = 0.15 #0.2 , 0.15
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)


        # Replay Memory
        self.buffer_size = 100000
        self.batch_size = 128 
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99 # 0.95 discount factor  ESTAVA 0.99
        self.tau = 0.001 # for soft update of target parameters
            
        # Score tracker and learning parameters
        self.score = 0.0
        self.best_score = -np.inf # Numpy - infinity
        self.count = 0
       
        # Reset the episode variables
        self.reset_episode()
        
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.t_reward = 0.0
        self.count = 0
        return state
       
    def step(self, action, reward, next_state, done):
        # Save Reward and experience
        self.memory.add(self.last_state, action, reward, next_state, done)
        
        self.t_reward += reward
        
        self.count += 1
        
        # Learn, if enough samples area available in memory
        if (len(self.memory) > self.batch_size):
            experiences = self.memory.sample()
            self.learn(experiences)

        # Updating the last state
        self.last_state = next_state
            
    def act(self, state):
        """Returns actions for given state(s) as per current policy.""" 
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        # Add some noise for exploration
        return list(action + self.noise.sample())

    def learn(self, experiences):
        '''Update the policy and value parameters using the tuples 'experiences''' 

        # Create arrays to reorganize all the batches separately
        states = np.vstack([b.state for b in experiences if b is not None])
        actions = np.array([b.action for b in experiences if b is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([b.reward for b in experiences if b is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([b.done for b in experiences if b is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([b.next_state for b in experiences if b is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        #* Compute Q Targets for current states and train critic model (local) 
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x = [states, actions], y=Q_targets)

        #* Train actor model (local)
        actions_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, actions_gradients, 1]) #* Custom training function

        #* Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
        # Update the score
        self.score = self.t_reward / float(self.count) if self.count > 0 else 0.0
        
        if self.score > self.best_score:
            self.best_score = self.score
    
    def soft_update(self, local_model, target_model):
        # Soft update model parameters
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        assert len(local_weights) == len(target_weights), "Local and target models parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) *  target_weights
        target_model.set_weights(new_weights) 