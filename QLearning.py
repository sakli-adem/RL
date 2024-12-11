import numpy as np
import random
# Model-Free Single-Agent Q-Learning approach
class QLearningAgent:
    def __init__(self, grid, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.q_table = np.zeros((self.rows, self.cols, 4))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [
            (-1, 0),  # up
            (0, 1),   # right
            (1, 0),   # down
            (0, -1)   # left
        ]
        self.action_names = ['Up', 'Right', 'Down', 'Left']

        # Positions initiales des matériaux
        self.materials_positions = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r, c] == 2]
        self.total_materials = len(self.materials_positions)
        # Sortie bas droite
        self.exit_state = (self.rows - 1, self.cols - 1)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Exploration
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploitation

    def get_next_state(self, state, action):
        next_row = state[0] + self.actions[action][0]
        next_col = state[1] + self.actions[action][1]

        if (0 <= next_row < self.rows and
            0 <= next_col < self.cols and
            self.grid[next_row, next_col] != 1):  # 1 = obstacle
            return (next_row, next_col), True
        return state, False

    def calculate_reward(self, state, collected):
        cell_value = self.grid[state[0], state[1]]

        # Si on est sur la sortie
        if state == self.exit_state:
            # Si tous les matériaux sont collectés, +20
            if len(collected) == self.total_materials:
                return 20
            else:
                return -1

        # Matériau non collecté
        if cell_value == 2 and state not in collected:
            return 10

        # Obstacle (devrait être évité)
        if cell_value == 1:
            return -10

        # Sinon déplacement normal
        return -1

    def train(self, start_state, max_episodes=2000):
        best_path = None
        best_reward = float('-inf')

        for episode in range(max_episodes):
            current_state = start_state
            episode_path = [current_state]
            total_reward = 0
            collected = []
            done = False
            steps = 0

            while not done and steps < 200:
                action = self.choose_action(current_state)
                next_state, is_valid = self.get_next_state(current_state, action)

                reward = self.calculate_reward(next_state, collected)
                total_reward += reward

                best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
                td_error = td_target - self.q_table[current_state[0], current_state[1], action]
                self.q_table[current_state[0], current_state[1], action] += self.lr * td_error

                current_state = next_state
                episode_path.append(current_state)

                # Ramassage de matériau
                if self.grid[current_state[0], current_state[1]] == 2 and current_state not in collected:
                    collected.append(current_state)

                # Condition d'arrêt : tous les matériaux collectés ET sortie atteinte
                if len(collected) == self.total_materials and current_state == self.exit_state:
                    done = True

                steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = episode_path

        return best_path, best_reward

    def print_q_table(self):
        print("Q-Table:")
        for r in range(self.rows):
            for c in range(self.cols):
                print(f"État ({r},{c}):")
                for a in range(4):
                    print(f"  {self.action_names[a]}: {self.q_table[r,c,a]:.2f}")
            print()



class QLearningAgent:
    def __init__(self, grid, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.q_table = np.zeros((self.rows, self.cols, 4))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [
            (-1, 0),  # up
            (0, 1),   # right
            (1, 0),   # down
            (0, -1)   # left
        ]
        self.action_names = ['Up', 'Right', 'Down', 'Left']

        # Positions initiales des matériaux
        self.materials_positions = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r, c] == 2]
        self.total_materials = len(self.materials_positions)
        # Sortie bas droite
        self.exit_state = (self.rows - 1, self.cols - 1)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Exploration
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploitation

    def get_next_state(self, state, action):
        next_row = state[0] + self.actions[action][0]
        next_col = state[1] + self.actions[action][1]

        if (0 <= next_row < self.rows and
            0 <= next_col < self.cols and
            self.grid[next_row, next_col] != 1):  # 1 = obstacle
            return (next_row, next_col), True
        return state, False

    def calculate_reward(self, state, collected):
        cell_value = self.grid[state[0], state[1]]

        # Si on est sur la sortie
        if state == self.exit_state:
            # Si tous les matériaux sont collectés, +20
            if len(collected) == self.total_materials:
                return 20
            else:
                return -1

        # Matériau non collecté
        if cell_value == 2 and state not in collected:
            return 10

        # Obstacle (devrait être évité)
        if cell_value == 1:
            return -10

        # Sinon déplacement normal
        return -1

    def train(self, start_state, max_episodes=2000):
        best_path = None
        best_reward = float('-inf')

        for episode in range(max_episodes):
            current_state = start_state
            episode_path = [current_state]
            total_reward = 0
            collected = []
            done = False
            steps = 0

            while not done and steps < 200:
                action = self.choose_action(current_state)
                next_state, is_valid = self.get_next_state(current_state, action)

                reward = self.calculate_reward(next_state, collected)
                total_reward += reward

                best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
                td_error = td_target - self.q_table[current_state[0], current_state[1], action]
                self.q_table[current_state[0], current_state[1], action] += self.lr * td_error

                current_state = next_state
                episode_path.append(current_state)

                # Ramassage de matériau
                if self.grid[current_state[0], current_state[1]] == 2 and current_state not in collected:
                    collected.append(current_state)

                # Condition d'arrêt : tous les matériaux collectés ET sortie atteinte
                if len(collected) == self.total_materials and current_state == self.exit_state:
                    done = True

                steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = episode_path

        return best_path, best_reward

    def print_q_table(self):
        print("Q-Table:")
        for r in range(self.rows):
            for c in range(self.cols):
                print(f"État ({r},{c}):")
                for a in range(4):
                    print(f"  {self.action_names[a]}: {self.q_table[r,c,a]:.2f}")
            print()



class QLearningAgentClean:
    def __init__(self, grid, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.q_table = np.zeros((self.rows, self.cols, 4))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [
            (-1, 0),
            (0, 1),
            (1, 0),
            (0, -1)
        ]
        self.action_names = ['Up', 'Right', 'Down', 'Left']

        # Identifiez toutes les cellules sales
        self.dirty_cells = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 2]
        self.cleaned = set()

        # Objectif final : en haut à droite
        self.top_right_exit = (0, self.cols - 1)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def get_next_state(self, state, action):
        next_row = state[0] + self.actions[action][0]
        next_col = state[1] + self.actions[action][1]

        if (0 <= next_row < self.rows and
            0 <= next_col < self.cols and
            self.grid[next_row, next_col] != 1):
            return (next_row, next_col), True
        return state, False

    def all_cells_cleaned(self):
        # Vérifie si toutes les cellules sales sont nettoyées
        return len(self.cleaned) == len(self.dirty_cells)

    def calculate_reward(self, state):
        # Si tout est nettoyé et qu'on atteint la sortie
        if self.all_cells_cleaned() and state == self.top_right_exit:
            return 50  # Récompense finale augmentée

        cell_value = self.grid[state[0], state[1]]

        # Cellule sale non nettoyée
        if cell_value == 2 and state not in self.cleaned:
            return 10  # Récompense pour nettoyage la première fois

        # Cellule sale déjà nettoyée (revisite)
        if cell_value == 2 and state in self.cleaned:
            return -5  # Pénalité pour revisiter augmentée

        # Cellule propre ou sans intérêt (0)
        return -1  # Déplacement normal

    def update_cleaning_status(self, state):
        if self.grid[state[0], state[1]] == 2 and state not in self.cleaned:
            self.cleaned.add(state)

    def train(self, start_state, max_episodes=2000):
        best_path = None
        best_reward = float('-inf')

        for episode in range(max_episodes):
            current_state = start_state
            episode_path = [current_state]
            total_reward = 0
            self.cleaned = set()
            done = False
            steps = 0

            while not done and steps < 500:
                action = self.choose_action(current_state)
                next_state, is_valid = self.get_next_state(current_state, action)

                reward = self.calculate_reward(next_state)
                total_reward += reward

                best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
                td_error = td_target - self.q_table[current_state[0], current_state[1], action]
                self.q_table[current_state[0], current_state[1], action] += self.lr * td_error

                current_state = next_state
                episode_path.append(current_state)

                # Mise à jour du statut de nettoyage
                self.update_cleaning_status(current_state)

                # Condition d'arrêt : toutes les cellules sales nettoyées + sur la sortie
                if self.all_cells_cleaned() and current_state == self.top_right_exit:
                    done = True

                steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = episode_path

        # Phase d'optimisation pour atteindre la sortie
        if best_path:
            last_cleaned_state = best_path[-1]
            best_path_to_exit, _ = self.optimize_path_to_exit(last_cleaned_state, max_episodes=1000)
            best_path = best_path[:-1] + best_path_to_exit

        return best_path, best_reward

    def optimize_path_to_exit(self, start_state, max_episodes=1000):
        best_path = None
        best_reward = float('-inf')

        for episode in range(max_episodes):
            current_state = start_state
            episode_path = [current_state]
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 500:
                action = self.choose_action(current_state)
                next_state, is_valid = self.get_next_state(current_state, action)

                reward = self.calculate_reward(next_state)
                total_reward += reward

                best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
                td_error = td_target - self.q_table[current_state[0], current_state[1], action]
                self.q_table[current_state[0], current_state[1], action] += self.lr * td_error

                current_state = next_state
                episode_path.append(current_state)

                # Condition d'arrêt : atteindre la sortie
                if current_state == self.top_right_exit:
                    done = True

                steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = episode_path

        return best_path, best_reward

    def print_q_table(self):
        print("Q-Table (Cleaning Env):")
        for r in range(self.rows):
            for c in range(self.cols):
                print(f"État ({r},{c}):")
                for a in range(4):
                    print(f"  {self.action_names[a]}: {self.q_table[r,c,a]:.2f}")
            print()

def visualize_path(grid, path, cleaning=False, stations=False):
    print("\nVisualisation du chemin:")
    grid_copy = grid.copy()

    for step, (row, col) in enumerate(path):
        step_grid = grid_copy.copy()
        step_grid[row, col] = 5

        print(f"\nÉtape {step}")
        for r in range(grid.shape[0]):
            row_str = ""
            for c in range(grid.shape[1]):
                cell_val = grid[r, c]
                if (r, c) == (row, col):
                    row_str += "A "
                elif cell_val == 1:
                    row_str += "X "
                else:
                    if cleaning:
                        # Env E
                        if cell_val == 2:
                            row_str += "U "
                        else:
                            row_str += "- "
                    elif stations:
                        # Env G : stations = 3
                        if cell_val == 3:
                            row_str += "S "
                        else:
                            row_str += "- "
                    else:
                        # Env 1
                        if cell_val == 2:
                            row_str += "M "
                        elif (r, c) == (grid.shape[0]-1, grid.shape[1]-1):
                            row_str += "E "
                        else:
                            row_str += "- "
            print(row_str)

class QLearningAgentStations:
    def __init__(self, grid, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.q_table = np.zeros((self.rows, self.cols, 4))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

        # Actions
        self.actions = [
            (-1, 0),  # up
            (0, 1),   # right
            (1, 0),   # down
            (0, -1)   # left
        ]
        self.action_names = ['Up', 'Right', 'Down', 'Left']

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def get_next_state(self, state, action):
        next_row = state[0] + self.actions[action][0]
        next_col = state[1] + self.actions[action][1]

        if (0 <= next_row < self.rows and
            0 <= next_col < self.cols and
            self.grid[next_row, next_col] != 1):
            return (next_row, next_col), True
        return state, False

    def calculate_reward(self, state):
        # Chaque pas = -1
        # Si station (3), +100 et fin
        if self.grid[state[0], state[1]] == 3:
            return 100
        return -1

    def train(self, start_state, max_episodes=2000):
        best_path = None
        best_reward = float('-inf')

        for episode in range(max_episodes):
            current_state = start_state
            episode_path = [current_state]
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 500:
                action = self.choose_action(current_state)
                next_state, is_valid = self.get_next_state(current_state, action)

                reward = self.calculate_reward(next_state)
                total_reward += reward

                best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
                td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
                td_error = td_target - self.q_table[current_state[0], current_state[1], action]
                self.q_table[current_state[0], current_state[1], action] += self.lr * td_error

                current_state = next_state
                episode_path.append(current_state)

                # Arrêter si station atteinte
                if self.grid[current_state[0], current_state[1]] == 3:
                    done = True

                steps += 1

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if total_reward > best_reward:
                best_reward = total_reward
                best_path = episode_path

        return best_path, best_reward

    def print_q_table(self):
        print("Q-Table (Stations Env):")
        for r in range(self.rows):
            for c in range(self.cols):
                print(f"État ({r},{c}):")
                for a in range(4):
                    print(f"  {self.action_names[a]}: {self.q_table[r,c,a]:.2f}")
            print()




def main():
    # ENV 1
    H = np.array([
        [0, 0, 2, 0],
        [2, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0]  
    ])
    start_state = (0, 0)
    agent = QLearningAgent(H)
    best_path, best_reward = agent.train(start_state)

    agent.print_q_table()
    visualize_path(H, best_path, cleaning=False)
    print("\nMeilleur chemin - Env 1:")
    for step, state in enumerate(best_path):
        print(f"Étape {step}: {state}")
    print(f"\nRécompense totale (Env 1): {best_reward}")

    # ENV E
    E = np.array([
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 1, 2, 2],
        [2, 2, 2, 2, 1]
    ])
    start_state_E = (0, 0)
    agent_clean = QLearningAgentClean(E)
    best_path_E, best_reward_E = agent_clean.train(start_state_E)
    agent_clean.print_q_table()
    visualize_path(E, best_path_E, cleaning=True)

    print("\nMeilleur chemin - Env E:")
    for step, state in enumerate(best_path_E):
        print(f"Étape {step}: {state}")
    print(f"\nRécompense totale (Env E): {best_reward_E}")

    # ENV G
    # 2 obstacles et 2 stations
    G = np.array([
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 3, 0]
    ])
    # Départ en bas à gauche
    start_state_G = (G.shape[0]-1, 0)
    agent_stations = QLearningAgentStations(G)
    best_path_G, best_reward_G = agent_stations.train(start_state_G)
    agent_stations.print_q_table()
    visualize_path(G, best_path_G, stations=True)

    print("\nMeilleur chemin - Env G:")
    for step, state in enumerate(best_path_G):
        print(f"Étape {step}: {state}")
    print(f"\nRécompense totale (Env G): {best_reward_G}")

if __name__ == "__main__":
    main()























