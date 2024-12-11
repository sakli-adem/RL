import numpy as np
import time

###################################
# Classe pour H et E (et potentiellement plusieurs cibles)
###################################
class ValueIterationCollectAll:
    def __init__(self, grid, env_type='H', gamma=0.9, theta=1e-4, reward_step=-1, reward_clean=10, penalty_revisit=-5):
        """
        env_type='H' ou 'E'

        grid: numpy array
            0 = vide
            1 = obstacle
            2 = objet d'intérêt : 
                - H: matériau (M)
                - E: cellule sale (U)

        Pour env_type='H':
            - "2" = matériau (M)
            - Collecter un matériau la première fois: +10
            - Revisite sans pénalité (le matériau disparaît une fois collecté)

        Pour env_type='E':
            - "2" = cellule sale (U)
            - Première fois: nettoyée (C) +10
            - Revisite sur une cellule déjà nettoyée (C): pénalité -5

        Dans les deux cas:
        - Chaque pas: -1
        - Atteindre la sortie avec tous les items collectés: +20

        gamma: facteur de discount
        theta: critère d'arrêt
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.env_type = env_type
        self.gamma = gamma
        self.theta = theta
        self.reward_step = reward_step
        self.reward_clean = reward_clean
        self.penalty_revisit = penalty_revisit

        # Les cibles sont toujours "2" dans H et E
        if self.env_type not in ['H', 'E']:
            raise ValueError("Pour cette classe, env_type doit être 'H' ou 'E'.")

        target_value = 2
        self.targets = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == target_value]
        self.num_targets = len(self.targets)
        
        self.target_index = {}
        for i, (tr, tc) in enumerate(self.targets):
            self.target_index[(tr,tc)] = i

        self.V = np.zeros((self.rows, self.cols, 2**self.num_targets))

        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]
        self.action_symbols = ['↑','→','↓','←']

        self.terminal_pos = (self.rows-1, self.cols-1)
    
    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != 1

    def is_terminal(self, r, c, mask):
        if (r,c) == self.terminal_pos:
            if mask == (2**self.num_targets - 1):
                return True
        return False

    def step(self, r, c, mask, action):
        nr = r + action[0]
        nc = c + action[1]

        if not self.is_valid(nr, nc):
            nr, nc = r, c

        reward = self.reward_step
        new_mask = mask
        cell_val = self.grid[nr, nc]

        if cell_val == 2:
            tgt_idx = self.target_index[(nr,nc)]
            already_collected = (mask & (1 << tgt_idx)) != 0
            if not already_collected:
                # Première fois
                new_mask = mask | (1 << tgt_idx)
                reward += self.reward_clean
            else:
                # Revisite
                if self.env_type == 'E':
                    reward += self.penalty_revisit
                # Pour H, rien de spécial

        if self.is_terminal(nr, nc, new_mask):
            reward += 20

        return nr, nc, new_mask, reward

    def value_iteration(self, max_iterations=1000):
        for i in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            for r in range(self.rows):
                for c in range(self.cols):
                    for mask in range(2**self.num_targets):
                        if self.is_terminal(r, c, mask) or self.grid[r,c] == 1:
                            continue
                        values_actions = []
                        for a_idx, action in enumerate(self.actions):
                            nr, nc, nmask, reward = self.step(r, c, mask, action)
                            values_actions.append(reward + self.gamma * self.V[nr, nc, nmask])
                        best_value = max(values_actions)
                        delta = max(delta, abs(best_value - self.V[r,c,mask]))
                        new_V[r,c,mask] = best_value
            self.V = new_V
            if delta < self.theta:
                print(f"Converged after {i} iterations.")
                break

    def extract_policy(self):
        policy = np.full((self.rows, self.cols, 2**self.num_targets), -1)
        for r in range(self.rows):
            for c in range(self.cols):
                for mask in range(2**self.num_targets):
                    if self.is_terminal(r,c,mask) or self.grid[r,c] == 1:
                        continue
                    values_actions = []
                    for a_idx, action in enumerate(self.actions):
                        nr, nc, nmask, reward = self.step(r, c, mask, action)
                        values_actions.append(reward + self.gamma * self.V[nr, nc, nmask])
                    best_action = np.argmax(values_actions)
                    policy[r,c,mask] = best_action
        return policy

    def print_values(self):
        print("Values (mask=0):")
        val_0 = self.V[:,:,0]
        print(val_0)

    def simulate_policy(self, policy, start_state=(0,0), delay=0.5):
        r, c = start_state
        mask = 0
        step_count = 0
        total_reward = 0.0

        while not self.is_terminal(r, c, mask):
            self.print_grid_with_agent(r, c, mask)
            a_idx = policy[r,c,mask]
            if a_idx == -1:
                print("Aucune action définie pour cet état, arrêt.")
                break
            action = self.actions[a_idx]
            nr, nc, nmask, reward = self.step(r, c, mask, action)
            total_reward += reward
            r, c, mask = nr, nc, nmask
            step_count += 1
            time.sleep(delay)

        self.print_grid_with_agent(r, c, mask)
        print(f"Arrivé à l'état terminal en {step_count} étapes.")
        print(f"Récompense totale obtenue: {total_reward}")

    def print_grid_with_agent(self, agent_r, agent_c, mask):
        print("\nÉtape:")
        collected_list = []
        for i in range(self.num_targets):
            if (mask & (1 << i)) != 0:
                collected_list.append(i)
        print(f"Indices collectés/nettoyés: {collected_list}")

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r,c) == (agent_r, agent_c):
                    row_str += "A "
                else:
                    cell_val = self.grid[r,c]
                    if cell_val == 1:
                        row_str += "X "
                    elif cell_val == 2:
                        tgt_idx = self.target_index[(r,c)]
                        already_done = (mask & (1 << tgt_idx)) != 0
                        if self.env_type == 'H':
                            # Matériau
                            if already_done:
                                row_str += ". "
                            else:
                                row_str += "M "
                        else:
                            # env_type='E'
                            # Cellule sale
                            if already_done:
                                row_str += "C "
                            else:
                                row_str += "U "
                    elif (r,c) == self.terminal_pos:
                        row_str += "E "
                    else:
                        row_str += ". "
            print(row_str)


###################################
# Classe pour G (une seule station, la plus proche)
###################################
class ValueIterationSingleTarget:
    def __init__(self, grid, start_state=(0,0), gamma=0.9, theta=1e-4, reward_step=-1, reward_collect=10):
        """
        Environnement G simplifié:
        - Une seule station (3) la plus proche du start.
        - L'agent doit juste atteindre cette station.
        
        grid:
            0 = vide
            1 = obstacle
            3 = station de recharge
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.gamma = gamma
        self.theta = theta
        self.reward_step = reward_step
        self.reward_collect = reward_collect
        self.start_state = start_state

        stations = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 3]
        if len(stations) == 0:
            raise ValueError("Aucune station (3) dans la grille G.")

        sr, sc = self.start_state
        distances = [abs(sr - r) + abs(sc - c) for (r,c) in stations]
        idx_min = np.argmin(distances)
        self.target = stations[idx_min]

        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]
        self.action_symbols = ['↑','→','↓','←']

        self.V = np.zeros((self.rows, self.cols))

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != 1

    def is_terminal(self, r, c):
        return (r,c) == self.target

    def step(self, r, c, action):
        nr = r + action[0]
        nc = c + action[1]
        if not self.is_valid(nr, nc):
            nr, nc = r, c

        reward = self.reward_step
        if self.is_terminal(nr, nc):
            reward += self.reward_collect

        return nr, nc, reward

    def value_iteration(self, max_iterations=1000):
        for i in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.is_terminal(r,c) or self.grid[r,c] == 1:
                        continue
                    values_actions = []
                    for action in self.actions:
                        nr, nc, rew = self.step(r,c,action)
                        values_actions.append(rew + self.gamma*self.V[nr,nc])
                    best_value = max(values_actions)
                    delta = max(delta, abs(best_value - self.V[r,c]))
                    new_V[r,c] = best_value
            self.V = new_V
            if delta < self.theta:
                print(f"Converged after {i} iterations.")
                break

    def extract_policy(self):
        policy = np.full((self.rows, self.cols), -1)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_terminal(r,c) or self.grid[r,c] == 1:
                    continue
                values_actions = []
                for a_idx, action in enumerate(self.actions):
                    nr, nc, rew = self.step(r,c,action)
                    values_actions.append(rew + self.gamma*self.V[nr,nc])
                best_action = np.argmax(values_actions)
                policy[r,c] = best_action
        return policy

    def simulate_policy(self, delay=0.5):
        r, c = self.start_state
        total_reward = 0.0
        step_count = 0
        policy = self.extract_policy()
        while not self.is_terminal(r,c):
            self.print_grid_with_agent(r,c)
            a_idx = policy[r,c]
            if a_idx == -1:
                print("Aucune action définie, arrêt.")
                break
            action = self.actions[a_idx]
            nr, nc, rew = self.step(r,c,action)
            total_reward += rew
            r,c = nr,nc
            step_count += 1
            time.sleep(delay)

        self.print_grid_with_agent(r,c)
        print(f"Arrivé à la station en {step_count} étapes.")
        print(f"Récompense totale: {total_reward}")

    def print_grid_with_agent(self, agent_r, agent_c):
        print("\nÉtape:")
        for rr in range(self.rows):
            row_str = ""
            for cc in range(self.cols):
                if (rr,cc) == (agent_r, agent_c):
                    row_str += "A "
                else:
                    cell_val = self.grid[rr,cc]
                    if cell_val == 1:
                        row_str += "X "
                    elif cell_val == 3:
                        # Station, la cible
                        if (rr,cc) == self.target:
                            row_str += "S "
                        else:
                            row_str += "s "
                    else:
                        row_str += ". "
            print(row_str)


###################################
# MAIN : Exemples d'exécution pour H, E et G
###################################
if __name__ == "__main__":
    # ENVIRONNEMENT H
    print("=== ENVIRONNEMENT H ===")
    H = np.array([
        [0, 0, 2, 0],
        [2, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    vi_H = ValueIterationCollectAll(H, env_type='H', gamma=0.9, theta=1e-4, reward_step=-1, reward_clean=10, penalty_revisit=-5)
    vi_H.value_iteration(max_iterations=1000)
    vi_H.print_values()
    policy_H = vi_H.extract_policy()
    vi_H.simulate_policy(policy_H, start_state=(0,0), delay=0.5)

    # ENVIRONNEMENT E
    print("\n=== ENVIRONNEMENT E ===")
    E = np.array([
        [0, 0, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 1, 2],
        [2, 2, 2, 2]
    ])
    # Pour E, on relâche les conditions de convergence pour accélérer
    vi_E = ValueIterationCollectAll(E, env_type='E', gamma=0.7, theta=1e-2, reward_step=-1, reward_clean=10, penalty_revisit=-5)
    vi_E.terminal_pos = (0, E.shape[1]-1)
    vi_E.value_iteration(max_iterations=200)
    vi_E.print_values()
    policy_E = vi_E.extract_policy()
    vi_E.simulate_policy(policy_E, start_state=(3,0), delay=0.5)

    # ENVIRONNEMENT G (une seule station la plus proche)
    print("\n=== ENVIRONNEMENT G ===")
    G = np.array([
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [0, 3, 0, 1],
        [0, 0, 0, 0]
    ])
    start_g = (G.shape[0]-1, 0)
    vi_G = ValueIterationSingleTarget(G, start_state=start_g, gamma=0.9, theta=1e-4, reward_step=-1, reward_collect=10)
    vi_G.value_iteration(max_iterations=1000)
    vi_G.simulate_policy(delay=0.5)
