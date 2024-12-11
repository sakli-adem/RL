import numpy as np
import random
# CTDE = Multi-Agent + Model-Free + Decentralized Execution
########################################
# Central Q-Learning Agent
########################################

class CentralQLearningAgent:
    def __init__(self, state_shape, num_actions=16, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        (self.rows1, self.cols1, self.rows2, self.cols2) = state_shape
        self.num_actions = num_actions 
        self.q_table = np.zeros((self.rows1, self.cols1, self.rows2, self.cols2, self.num_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

    def state_to_indices(self, pos1, pos2):
        return (pos1[0], pos1[1], pos2[0], pos2[1])

    def choose_action(self, pos1, pos2):
        s = self.state_to_indices(pos1, pos2)
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            return np.argmax(self.q_table[s])

    def update_q_value(self, pos1, pos2, action, reward, next_pos1, next_pos2):
        s = self.state_to_indices(pos1, pos2)
        s_next = self.state_to_indices(next_pos1, next_pos2)
        best_next_action = np.argmax(self.q_table[s_next])
        td_target = reward + self.gamma * self.q_table[s_next][best_next_action]
        td_error = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def decode_action(joint_action):
    a1 = joint_action //4
    a2 = joint_action %4
    return a1, a2

########################################
# Environment H (CTDE)
########################################

class MultiAgentEnvironmentH_CTDE:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.actions_single = [(-1,0),(0,1),(1,0),(0,-1)]
        self.materials_positions = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 2]
        self.total_materials = len(self.materials_positions)
        assert self.total_materials == 2, "H: Il doit y avoir 2 matériaux."
        self.exit_state = (self.rows-1, self.cols-1)
        self.agent_got_material = [False, False]
        self.collected = set()

    def reset(self):
        free_positions = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 0]
        assert len(free_positions) >= 2, "H: Pas assez de cases libres."
        start_positions = random.sample(free_positions, 2)
        self.agent_positions = [start_positions[0], start_positions[1]]
        self.collected = set()
        self.agent_got_material = [False, False]
        self.targets = []
        for pos in self.agent_positions:
            distances = [abs(pos[0]-m[0]) + abs(pos[1]-m[1]) for m in self.materials_positions]
            min_idx = np.argmin(distances)
            self.targets.append(self.materials_positions[min_idx])
        return self.agent_positions[0], self.agent_positions[1]

    def step(self, joint_action):
        a1,a2 = decode_action(joint_action)
        rewards = [0,0]
        pos1, pos2 = self.agent_positions

        # agent 1
        if self.agent_got_material[0] and pos1 == self.exit_state:
            new_pos1 = pos1
        else:
            nr1 = pos1[0] + self.actions_single[a1][0]
            nc1 = pos1[1] + self.actions_single[a1][1]
            if (0 <= nr1 < self.rows and 0 <= nc1 < self.cols and self.grid[nr1,nc1] != 1):
                new_pos1 = (nr1,nc1)
            else:
                new_pos1 = pos1
                rewards[0] += -10

        # agent 2
        if self.agent_got_material[1] and pos2 == self.exit_state:
            new_pos2 = pos2
        else:
            nr2 = pos2[0] + self.actions_single[a2][0]
            nc2 = pos2[1] + self.actions_single[a2][1]
            if (0 <= nr2 < self.rows and 0 <= nc2 < self.cols and self.grid[nr2,nc2] != 1):
                new_pos2 = (nr2,nc2)
            else:
                new_pos2 = pos2
                rewards[1] += -10

        self.agent_positions = [new_pos1, new_pos2]

        # Récompenses
        for i, pos in enumerate(self.agent_positions):
            cell_value = self.grid[pos[0], pos[1]]
            if not self.agent_got_material[i]:
                target = self.targets[i]
                if cell_value == 2 and pos == target and pos not in self.collected:
                    self.collected.add(pos)
                    self.agent_got_material[i] = True
                    rewards[i] += 10
                else:
                    if pos != self.exit_state:
                        rewards[i] += -1
            else:
                if pos != self.exit_state:
                    rewards[i] += -1

        done = False
        if all(self.agent_got_material) and self.agent_positions[0] == self.exit_state and self.agent_positions[1] == self.exit_state:
            done = True
            rewards[0] += 20
            rewards[1] += 20

        total_reward = sum(rewards)
        return self.agent_positions[0], self.agent_positions[1], total_reward, done

def train_multiagent_H_CTDE(grid, max_episodes=500, max_steps=200):
    env = MultiAgentEnvironmentH_CTDE(grid)
    state_shape = (env.rows, env.cols, env.rows, env.cols)
    agent = CentralQLearningAgent(state_shape)

    best_reward = float('-inf')
    best_data = None
    for episode in range(max_episodes):
        pos1, pos2 = env.reset()
        total_reward = 0
        path1 = [pos1]
        path2 = [pos2]
        done = False
        for step in range(max_steps):
            action = agent.choose_action(pos1, pos2)
            next_pos1, next_pos2, reward, done = env.step(action)
            agent.update_q_value(pos1, pos2, action, reward, next_pos1, next_pos2)
            pos1, pos2 = next_pos1, next_pos2
            total_reward += reward
            path1.append(pos1)
            path2.append(pos2)
            if done:
                break
        agent.decay_epsilon()
        if total_reward > best_reward:
            best_reward = total_reward
            best_data = ([path1,path2], best_reward)
    return best_data, agent


########################################
# Environment E (CTDE)
########################################

class MultiAgentEnvironmentE_CTDE:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.actions_single = [(-1,0),(0,1),(1,0),(0,-1)]
        self.dirty_cells = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 2]
        self.total_dirty = len(self.dirty_cells)
        self.exit_state = (0, self.cols-1)
        self.cleaned = set()

    def all_cleaned(self):
        return len(self.cleaned) == self.total_dirty

    def reset(self):
        free_positions = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 0]
        assert len(free_positions) >= 2, "E_CTDE: Pas assez de cases libres."
        start_positions = random.sample(free_positions, 2)
        self.agent_positions = [start_positions[0], start_positions[1]]
        self.cleaned = set()
        return self.agent_positions[0], self.agent_positions[1]

    def step(self, joint_action):
        a1, a2 = decode_action(joint_action)
        rewards = [0,0]
        pos1, pos2 = self.agent_positions

        # Agent 1
        if self.all_cleaned() and pos1 == self.exit_state:
            new_pos1 = pos1
        else:
            nr1 = pos1[0] + self.actions_single[a1][0]
            nc1 = pos1[1] + self.actions_single[a1][1]
            if (0 <= nr1 < self.rows and 0 <= nc1 < self.cols and self.grid[nr1,nc1] != 1):
                new_pos1 = (nr1,nc1)
            else:
                new_pos1 = pos1
                rewards[0] += -10

        # Agent 2
        if self.all_cleaned() and pos2 == self.exit_state:
            new_pos2 = pos2
        else:
            nr2 = pos2[0] + self.actions_single[a2][0]
            nc2 = pos2[1] + self.actions_single[a2][1]
            if (0 <= nr2 < self.rows and 0 <= nc2 < self.cols and self.grid[nr2,nc2] != 1):
                new_pos2 = (nr2,nc2)
            else:
                new_pos2 = pos2
                rewards[1] += -10

        self.agent_positions = [new_pos1, new_pos2]

        # Rewards
        for i, pos in enumerate(self.agent_positions):
            cell_value = self.grid[pos[0], pos[1]]
            if cell_value == 2 and pos not in self.cleaned:
                self.cleaned.add(pos)
                rewards[i] += 10
            else:
                if cell_value == 2 and pos in self.cleaned:
                    rewards[i] += -5
                else:
                    if pos != self.exit_state:
                        rewards[i] += -1

        done = False
        if self.all_cleaned():
            if self.agent_positions[0] == self.exit_state and self.agent_positions[1] == self.exit_state:
                done = True
                rewards[0] += 50
                rewards[1] += 50

        total_reward = sum(rewards)
        return self.agent_positions[0], self.agent_positions[1], total_reward, done

def train_multiagent_E_CTDE(grid, max_episodes=500, max_steps=300):
    env = MultiAgentEnvironmentE_CTDE(grid)
    state_shape = (env.rows, env.cols, env.rows, env.cols)
    agent = CentralQLearningAgent(state_shape)

    best_reward = float('-inf')
    best_data = None
    for episode in range(max_episodes):
        pos1, pos2 = env.reset()
        total_reward = 0
        path1 = [pos1]
        path2 = [pos2]
        done = False
        for step in range(max_steps):
            action = agent.choose_action(pos1, pos2)
            next_pos1, next_pos2, reward, done = env.step(action)
            agent.update_q_value(pos1, pos2, action, reward, next_pos1, next_pos2)
            pos1, pos2 = next_pos1, next_pos2
            total_reward += reward
            path1.append(pos1)
            path2.append(pos2)
            if done:
                break
        agent.decay_epsilon()
        if total_reward > best_reward:
            best_reward = total_reward
            best_data = ([path1,path2], best_reward)
    return best_data, agent


########################################
# Environment G (CTDE) - avec condition: agents sur 2 stations distinctes
########################################


class MultiAgentEnvironmentG_CTDE:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.actions_single = [(-1,0),(0,1),(1,0),(0,-1)]
        self.stations = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 3]
        self.agents_done = [False, False]

    def reset(self):
        bottom_row = self.rows-1
        candidates = [(bottom_row,0),(bottom_row,1)]
        free_candidates = [pos for pos in candidates if self.grid[pos[0], pos[1]] == 0]
        if len(free_candidates) < 1:
            raise ValueError("G_CTDE: Impossible de placer les agents.")
        start_position = free_candidates[0]

        self.agent_positions = [start_position, start_position]
        self.agents_done = [False, False]
        return self.agent_positions[0], self.agent_positions[1]

    def step(self, joint_action):
        a1,a2 = decode_action(joint_action)
        rewards = [0,0]
        pos1, pos2 = self.agent_positions

        # Agent 1
        if self.agents_done[0]:
            new_pos1 = pos1
        else:
            nr1 = pos1[0] + self.actions_single[a1][0]
            nc1 = pos1[1] + self.actions_single[a1][1]
            if (0 <= nr1 < self.rows and 0 <= nc1 < self.cols and self.grid[nr1,nc1] != 1):
                new_pos1 = (nr1,nc1)
            else:
                new_pos1 = pos1
                rewards[0] += -10

        # Agent 2
        if self.agents_done[1]:
            new_pos2 = pos2
        else:
            nr2 = pos2[0] + self.actions_single[a2][0]
            nc2 = pos2[1] + self.actions_single[a2][1]
            if (0 <= nr2 < self.rows and 0 <= nc2 < self.cols and self.grid[nr2,nc2] != 1):
                new_pos2 = (nr2,nc2)
            else:
                new_pos2 = pos2
                rewards[1] += -10

        self.agent_positions = [new_pos1, new_pos2]

        for i, pos in enumerate(self.agent_positions):
            if not self.agents_done[i]:
                if self.grid[pos[0], pos[1]] == 3:
                    if pos == self.agent_positions[1-i]:
                        rewards[i] += -50  # Penalty for going to the same station
                    else:
                        rewards[i] += 100
                        self.agents_done[i] = True
                else:
                    rewards[i] += -1

        done = False
        if all(self.agents_done):
            done = True

        total_reward = sum(rewards)
        return self.agent_positions[0], self.agent_positions[1], total_reward, done

def train_multiagent_G_CTDE(grid, max_episodes=500, max_steps=300):
    env = MultiAgentEnvironmentG_CTDE(grid)
    state_shape = (env.rows, env.cols, env.rows, env.cols)
    agent = CentralQLearningAgent(state_shape)

    best_reward = float('-inf')
    best_data = None
    for episode in range(max_episodes):
        pos1, pos2 = env.reset()
        total_reward = 0
        path1 = [pos1]
        path2 = [pos2]
        done = False
        for step in range(max_steps):
            action = agent.choose_action(pos1, pos2)
            next_pos1, next_pos2, reward, done = env.step(action)
            agent.update_q_value(pos1, pos2, action, reward, next_pos1, next_pos2)
            pos1, pos2 = next_pos1, next_pos2
            total_reward += reward
            path1.append(pos1)
            path2.append(pos2)
            if done:
                break
        agent.decay_epsilon()
        if total_reward > best_reward:
            best_reward = total_reward
            best_data = ([path1,path2], best_reward)
    return best_data, agent

########################################
# Visualisation
########################################

def visualize_ctde_paths(grid, paths, env_type='H'):
    print(f"\nVisualisation du chemin final ({env_type}) - CTDE:\n")
    max_steps = max(len(paths[0]), len(paths[1]))
    for step in range(max_steps):
        print(f"Étape {step}:")
        pos1 = paths[0][step] if step < len(paths[0]) else paths[0][-1]
        pos2 = paths[1][step] if step < len(paths[1]) else paths[1][-1]
        for r in range(grid.shape[0]):
            row_str = ""
            for c in range(grid.shape[1]):
                if (r,c) == pos1 and (r,c) == pos2:
                    row_str += "X "
                elif (r,c) == pos1:
                    row_str += "A1"
                elif (r,c) == pos2:
                    row_str += "A2"
                else:
                    val = grid[r,c]
                    if val == 0:
                        row_str += "- "
                    elif val == 1:
                        row_str += "X "
                    elif val == 2 and env_type=='H':
                        row_str += "M "
                    elif val == 2 and env_type=='E':
                        row_str += "U "
                    elif val == 3 and env_type=='G':
                        row_str += "S "
                    else:
                        # sortie ou inconnu
                        if env_type=='H' and (r,c) == (grid.shape[0]-1, grid.shape[1]-1):
                            row_str += "E "
                        elif env_type=='E' and (r,c) == (0,grid.shape[1]-1):
                            row_str += "E "
                        else:
                            row_str += "- "
            print(row_str)
        print()

if __name__ == "__main__":
    # Grille H CTDE
    H = np.array([
        [0, 0, 2, 0],
        [0, 0, 0, 1],
        [0, 2, 0, 0],
        [0, 1, 0, 0]
    ])
    (paths_h_ctde, best_reward_h_ctde), _ = train_multiagent_H_CTDE(H)
    print("\nMeilleure récompense totale obtenue dans H (MARL, CTDE):", best_reward_h_ctde)
    visualize_ctde_paths(H, paths_h_ctde, 'H')

    # Grille E CTDE
    E = np.array([
        [0, 0, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 1, 2, 2],
        [2, 2, 2, 2, 1]
    ])
    (paths_e_ctde, best_reward_e_ctde), _ = train_multiagent_E_CTDE(E)
    print("\nMeilleure récompense totale obtenue dans E (MARL, CTDE):", best_reward_e_ctde)
    visualize_ctde_paths(E, paths_e_ctde, 'E')

    # Grille G CTDE
    G = np.array([
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 3, 0]
    ])
    (paths_g_ctde, best_reward_g_ctde), _ = train_multiagent_G_CTDE(G)
    print("\nMeilleure récompense totale obtenue dans G (MARL, CTDE):", best_reward_g_ctde)
    visualize_ctde_paths(G, paths_g_ctde, 'G')