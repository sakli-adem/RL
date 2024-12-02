import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Define the neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Agent class for GridH using DQN
class AgentH:
    def __init__(self, initial_position, grid, agent_id, input_size, output_size,
                 device='cpu', alpha=0.005, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.position = initial_position
        self.grid = grid
        self.agent_id = agent_id

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Neural Network
        self.device = torch.device(device)
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

        # Experience Replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Movement tracking
        self.total_reward = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _get_action_neighbors(self, action):
        x, y = action
        neighbors = [
            (x-1, y),   # Up
            (x+1, y),   # Down
            (x, y-1),   # Left
            (x, y+1)    # Right
        ]
        return [n for n in neighbors if 0 <= n[0] < self.grid.size[0] and 0 <= n[1] < self.grid.size[1] and n not in self.grid.bad_states]

    def get_state(self):
        # Enhanced state representation with more directional information
        state = np.zeros((self.grid.size[0], self.grid.size[1], 4))

        # Agent position
        state[self.position[0], self.position[1], 0] = 1

        # Material positions and distances
        for mat in self.grid.materials:
            state[mat[0], mat[1], 1] = 1
            dist = abs(mat[0] - self.position[0]) + abs(mat[1] - self.position[1])
            state[mat[0], mat[1], 2] = 1 / (dist + 1)

        # Obstacle positions
        for obs in self.grid.obstacles:
            state[obs[0], obs[1], 3] = 1

        return state.flatten()

    def choose_action(self, state):
        # Improved action selection with exploration strategy
        if random.random() < self.epsilon:
            # Explore: Prefer neighboring cells near materials
            candidates = self._get_action_neighbors(self.position)
            if candidates:
                return min(candidates,
                           key=lambda x: self._manhattan_distance(x, min(self.grid.materials,
                                                                   key=lambda m: self._manhattan_distance(x, m))))
            else:
                return self.position

        # Exploit: Use neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            valid_actions = [a for a in self.grid.states if a not in self.grid.bad_states]
            valid_indices = [self.grid.states.index(a) for a in valid_actions]
            valid_q_values = q_values[0][valid_indices]
            best_action_idx = valid_indices[torch.argmax(valid_q_values).item()]
            return self.grid.states[best_action_idx]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def perform_action(self, action):
        reward = 0

        # Obstacle penalty
        if action in self.grid.bad_states:
            return -100  # Significant penalty for hitting an obstacle

        # Material collection
        if action in self.grid.materials:
            self.grid.materials.remove(action)
            reward = 50  # High reward for collecting material

        # Movement mechanics
        self.grid.grid[self.position[0]][self.position[1]] = ''
        self.position = action
        self.grid.grid[self.position[0]][self.position[1]] = self.agent_id

        # Distance-based reward
        if self.grid.materials:
            nearest_material = min(self.grid.materials,
                                   key=lambda m: self._manhattan_distance(self.position, m))
            distance_to_nearest = self._manhattan_distance(self.position, nearest_material)
            reward -= distance_to_nearest * 0.5  # Penalize being far from materials

        # Exploration decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return reward

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Predict Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-value
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.criterion(current_q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def move(self):
        # Get current state
        state = self.get_state()

        # Choose and perform action
        action = self.choose_action(state)
        reward = self.perform_action(action)

        # Get next state
        next_state = self.get_state()

        # Store experience
        self.replay_buffer.add(state, self.grid.states.index(action), reward, next_state, 0)

        # Train the model
        self.train()

        self.total_reward += reward
        return reward

    def move_to_target(self, target):
        # Get current state
        state = self.get_state()

        # Choose and perform action
        action = self.choose_action_to_target(state, target)
        reward = self.perform_action(action)

        # Get next state
        next_state = self.get_state()

        # Store experience
        self.replay_buffer.add(state, self.grid.states.index(action), reward, next_state, 0)

        # Train the model
        self.train()

        self.total_reward += reward
        return reward

    def choose_action_to_target(self, state, target):
        # Explore: Prefer neighboring cells near the target
        candidates = self._get_action_neighbors(self.position)
        if candidates:
            return min(candidates,
                       key=lambda x: self._manhattan_distance(x, target))
        else:
            return self.position

# Agent class for GridE using DQN
class AgentE:
    def __init__(self, initial_position, grid, agent_id, input_size, output_size,
                 device='cpu', alpha=0.005, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.position = initial_position
        self.grid = grid
        self.agent_id = agent_id

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Neural Network
        self.device = torch.device(device)
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

        # Experience Replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Movement tracking
        self.total_reward = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _get_action_neighbors(self, action):
        x, y = action
        neighbors = [
            (x-1, y),   # Up
            (x+1, y),   # Down
            (x, y-1),   # Left
            (x, y+1)    # Right
        ]
        return [n for n in neighbors if 0 <= n[0] < self.grid.size[0] and 0 <= n[1] < self.grid.size[1] and n not in self.grid.bad_states]

    def get_state(self):
        # Enhanced state representation with more directional information
        state = np.zeros((self.grid.size[0], self.grid.size[1], 3))

        # Agent position
        state[self.position[0], self.position[1], 0] = 1

        # Unclean positions
        for x in range(self.grid.size[0]):
            for y in range(self.grid.size[1]):
                if self.grid.grid[x][y] == 'U':
                    state[x, y, 1] = 1

        # Obstacle positions
        for obs in self.grid.obstacles:
            state[obs[0], obs[1], 2] = 1

        return state.flatten()

    def choose_action(self, state):
        # Improved action selection with exploration strategy
        if random.random() < self.epsilon:
            # Explore: Prefer neighboring cells near unclean cells
            candidates = self._get_action_neighbors(self.position)
            if candidates:
                return min(candidates,
                           key=lambda x: self._manhattan_distance(x, min([(i, j) for i in range(self.grid.size[0]) for j in range(self.grid.size[1]) if self.grid.grid[i][j] == 'U'],
                                                                   key=lambda u: self._manhattan_distance(x, u))))
            else:
                return self.position

        # Exploit: Use neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            valid_actions = [a for a in self.grid.states if a not in self.grid.bad_states]
            valid_indices = [self.grid.states.index(a) for a in valid_actions]
            valid_q_values = q_values[0][valid_indices]
            best_action_idx = valid_indices[torch.argmax(valid_q_values).item()]
            return self.grid.states[best_action_idx]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def perform_action(self, action):
        reward = 0

        # Obstacle penalty
        if action in self.grid.bad_states:
            return -100  # Significant penalty for hitting an obstacle

        # Cleaning action
        if self.grid.grid[action[0]][action[1]] == 'U':
            self.grid.grid[action[0]][action[1]] = 'C'  # Mark as cleaned
            reward = 50  # High reward for cleaning

        # Movement mechanics
        if self.position != action:
            self.grid.grid[self.position[0]][self.position[1]] = 'C' if self.grid.grid[self.position[0]][self.position[1]] == 'U' else 'C'
        self.position = action
        self.grid.grid[self.position[0]][self.position[1]] = self.agent_id

        # Exploration decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return reward

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Predict Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-value
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.criterion(current_q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def move(self):
        # Get current state
        state = self.get_state()

        # Choose and perform action
        action = self.choose_action(state)
        reward = self.perform_action(action)

        # Get next state
        next_state = self.get_state()

        # Store experience
        self.replay_buffer.add(state, self.grid.states.index(action), reward, next_state, 0)

        # Train the model
        self.train()

        self.total_reward += reward
        return reward

    def move_to_target(self, target):
        # Get current state
        state = self.get_state()

        # Choose and perform action
        action = self.choose_action_to_target(state, target)
        reward = self.perform_action(action)

        # Get next state
        next_state = self.get_state()

        # Store experience
        self.replay_buffer.add(state, self.grid.states.index(action), reward, next_state, 0)

        # Train the model
        self.train()

        self.total_reward += reward
        return reward

    def choose_action_to_target(self, state, target):
        # Explore: Prefer neighboring cells near the target
        candidates = self._get_action_neighbors(self.position)
        if candidates:
            return min(candidates,
                       key=lambda x: self._manhattan_distance(x, target))
        else:
            return self.position

# Agent class for GridG using DQN
class AgentG:
    def __init__(self, initial_position, grid, agent_id, input_size, output_size,
                 device='cpu', alpha=0.005, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.position = initial_position
        self.grid = grid
        self.agent_id = agent_id
        self.found_station = False  # Flag to indicate if the agent has found a station

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Neural Network
        self.device = torch.device(device)
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

        # Experience Replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Movement tracking
        self.total_reward = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _get_action_neighbors(self, action):
        x, y = action
        neighbors = [
            (x-1, y),   # Up
            (x+1, y),   # Down
            (x, y-1),   # Left
            (x, y+1)    # Right
        ]
        return [n for n in neighbors if 0 <= n[0] < self.grid.size[0] and 0 <= n[1] < self.grid.size[1] and n not in self.grid.bad_states]

    def get_state(self):
        # Enhanced state representation with more directional information
        state = np.zeros((self.grid.size[0], self.grid.size[1], 3))

        # Agent position
        state[self.position[0], self.position[1], 0] = 1

        # Station positions
        for station in self.grid.stations:
            state[station[0], station[1], 1] = 1

        # Obstacle positions
        for obs in self.grid.obstacles:
            state[obs[0], obs[1], 2] = 1

        return state.flatten()

    def choose_action(self, state):
        # Improved action selection with exploration strategy
        if random.random() < self.epsilon:
            # Explore: Prefer neighboring cells near stations
            candidates = self._get_action_neighbors(self.position)
            if candidates:
                if self.grid.stations:
                    return min(candidates,
                               key=lambda x: self._manhattan_distance(x, min(self.grid.stations,
                                                                       key=lambda s: self._manhattan_distance(x, s))))
                else:
                    # If no stations are left, choose a random valid action
                    return random.choice(candidates)
            else:
                return self.position

        # Exploit: Use neural network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            valid_actions = [a for a in self.grid.states if a not in self.grid.bad_states]
            valid_indices = [self.grid.states.index(a) for a in valid_actions]
            valid_q_values = q_values[0][valid_indices]
            best_action_idx = valid_indices[torch.argmax(valid_q_values).item()]
            return self.grid.states[best_action_idx]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def perform_action(self, action):
        reward = 0

        # Obstacle penalty
        if action in self.grid.bad_states:
            return -100  # Significant penalty for hitting an obstacle

        # Station collection
        if action in self.grid.stations:
            self.grid.stations.remove(action)
            reward = 50  # High reward for reaching the station
            self.found_station = True  # Set the flag to indicate the station is found

        # Movement mechanics
        self.grid.grid[self.position[0]][self.position[1]] = ''
        self.position = action
        self.grid.grid[self.position[0]][self.position[1]] = self.agent_id

        # Exploration decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return reward

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Predict Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-value
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.criterion(current_q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def move(self):
        # Get current state
        state = self.get_state()

        # Choose and perform action
        action = self.choose_action(state)
        reward = self.perform_action(action)

        # Get next state
        next_state = self.get_state()

        # Store experience
        self.replay_buffer.add(state, self.grid.states.index(action), reward, next_state, 0)

        # Train the model
        self.train()

        self.total_reward += reward
        return reward

    def move_to_target(self, target):
        # Get current state
        state = self.get_state()

        # Choose and perform action
        action = self.choose_action_to_target(state, target)
        reward = self.perform_action(action)

        # Get next state
        next_state = self.get_state()

        # Store experience
        self.replay_buffer.add(state, self.grid.states.index(action), reward, next_state, 0)

        # Train the model
        self.train()

        self.total_reward += reward
        return reward

    def choose_action_to_target(self, state, target):
        # Explore: Prefer neighboring cells near the target
        candidates = self._get_action_neighbors(self.position)
        if candidates:
            return min(candidates,
                       key=lambda x: self._manhattan_distance(x, target))
        else:
            return self.position

# GridH class
class GridH:
    def __init__(self, size, num_agents, num_materials, num_obstacles):
        self.size = size
        self.grid = [['' for _ in range(size[1])] for _ in range(size[0])]
        self.agents = []
        self.materials = []
        self.obstacles = []
        self.states = [(x, y) for x in range(size[0]) for y in range(size[1])]
        self.bad_states = set()

        # Place obstacles
        for _ in range(num_obstacles):
            self._place_item('O', self.obstacles)

        # Place materials
        for _ in range(num_materials):
            self._place_item('M', self.materials)

        # Place agents
        for i in range(num_agents):
            agent_position = self._place_item(f'A{i}', None)
            input_size = size[0] * size[1] * 4
            output_size = len(self.states)
            self.agents.append(AgentH(agent_position, self, f'A{i}',
                                       input_size, output_size))

    def _place_item(self, symbol, collection):
        while True:
            x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
            if not self.grid[x][y]:
                self.grid[x][y] = symbol
                if collection is not None:
                    collection.append((x, y))
                if symbol == 'O':
                    self.bad_states.add((x, y))
                return (x, y)

    def display(self):
        print("\nGrid H:")
        for row in self.grid:
            print(' '.join(cell if cell else '.' for cell in row))

# GridE class
class GridE:
    def __init__(self, size, num_obstacles):
        self.size = size
        self.grid = [['U' for _ in range(size[1])] for _ in range(size[0])]
        self.states = [(x, y) for x in range(size[0]) for y in range(size[1])]
        self.obstacles = []
        self.bad_states = set()

        # Place obstacles
        for _ in range(num_obstacles):
            self._place_item('O', self.obstacles)

    def _place_item(self, symbol, collection):
        while True:
            x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
            if self.grid[x][y] == 'U':
                self.grid[x][y] = symbol
                if collection is not None:
                    collection.append((x, y))
                if symbol == 'O':
                    self.bad_states.add((x, y))
                return (x, y)

    def display(self):
        print("\nGrid E:")
        for row in self.grid:
            print(' '.join(cell for cell in row))

# GridG class
# GridG class with obstacles
class GridG:
    def __init__(self, size, num_agents, num_stations, num_obstacles):
        self.size = size
        self.grid = [['' for _ in range(size[1])] for _ in range(size[0])]
        self.agents = []
        self.stations = []
        self.obstacles = []
        self.states = [(x, y) for x in range(size[0]) for y in range(size[1])]
        self.bad_states = set()

        # Place obstacles
        for _ in range(num_obstacles):
            self._place_item('O', self.obstacles)

        # Place stations
        for _ in range(num_stations):
            self._place_item('R', self.stations)

    def _place_item(self, symbol, collection):
        while True:
            x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
            if not self.grid[x][y]:
                self.grid[x][y] = symbol
                if collection is not None:
                    collection.append((x, y))
                if symbol == 'O':
                    self.bad_states.add((x, y))
                return (x, y)

    def display(self):
        print("\nGrid G:")
        for row in self.grid:
            print(' '.join(cell if cell else '.' for cell in row))

def run_simulation(size_x=4, size_y=4, num_agents=1, episodes=1000):
    # Prompt the user for the number of obstacles and materials
    num_obstacles = int(input("Enter the number of obstacles: "))
    num_materials = int(input("Enter the number of materials: "))

    grid_h = GridH(size=(size_x, size_y), num_agents=num_agents,
                   num_materials=num_materials, num_obstacles=num_obstacles)
    grid_e = GridE(size=(size_x, size_y), num_obstacles=num_obstacles)
    grid_g = GridG(size=(size_x, size_y), num_agents=num_agents, num_stations=1, num_obstacles=num_obstacles)

    cumulative_reward = 0

    for episode in range(episodes):
        print(f"Episode {episode}")
        grid_h.display()

        episode_rewards = []
        for agent in grid_h.agents:
            reward = agent.move()
            episode_rewards.append(reward)
            print(f"Agent {agent.agent_id}, Reward: {reward}")

        # Update cumulative reward
        cumulative_reward += sum(episode_rewards)

        if not grid_h.materials:
            print("All materials collected!")
            print("Final Grid H State:")
            grid_h.display()

            # Move agent to bottom-right corner
            target = (size_x - 1, size_y - 1)
            while agent.position != target:
                reward = agent.move_to_target(target)
                episode_rewards.append(reward)
                print(f"Agent {agent.agent_id}, Reward: {reward}")
                grid_h.display()

            # Transition to Grid E
            agent_e = AgentE((0, 0), grid_e, agent.agent_id,
                              size_x * size_y * 3, len(grid_e.states))
            grid_e.grid[0][0] = agent.agent_id
            print("Transition to Grid E:")
            grid_e.display()

            # Start cleaning in Grid E
            while any('U' in row for row in grid_e.grid):
                reward = agent_e.move()
                episode_rewards.append(reward)
                print(f"Agent {agent_e.agent_id}, Reward: {reward}")
                grid_e.display()

            # Move to top-right corner
            target = (0, size_y - 1)
            while agent_e.position != target:
                reward = agent_e.move_to_target(target)
                episode_rewards.append(reward)
                print(f"Agent {agent_e.agent_id}, Reward: {reward}")
                grid_e.display()

            # Transition to Grid G
            agent_g = AgentG((size_x - 1, 0), grid_g, agent.agent_id,
                              size_x * size_y * 3, len(grid_g.states))
            grid_g.grid[size_x - 1][0] = agent.agent_id
            print("Transition to Grid G:")
            grid_g.display()

            # Start searching for the station in Grid G
            while not agent_g.found_station:
                reward = agent_g.move()
                episode_rewards.append(reward)
                print(f"Agent {agent_g.agent_id}, Reward: {reward}")
                grid_g.display()

            break

        # Print cumulative reward
        print(f"Cumulative Total Reward: {cumulative_reward}")

    # Print the final cumulative reward
    print(f"Final Cumulative Total Reward: {cumulative_reward}")

    return grid_h, grid_e, grid_g

# Example usage
if __name__ == "__main__":
    final_grid_h, final_grid_e, final_grid_g = run_simulation()
