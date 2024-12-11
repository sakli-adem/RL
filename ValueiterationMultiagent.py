import numpy as np

class MultiAgentValueIteration:
    def __init__(self, grid, gamma=0.9, theta=1e-4, reward_step=-1, reward_collect=10, penalty_revisit=-5):
        """
        Multi-agent Value Iteration for a scenario similar to H:
        - Two agents must each collect one piece of material (2 in total).
        - After collecting all materials, both must go to exit.

        grid: numpy array
            0 = empty
            1 = obstacle
            2 = material

        We assume exactly 2 materials for simplicity.
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.gamma = gamma
        self.theta = theta
        self.reward_step = reward_step
        self.reward_collect = reward_collect
        self.penalty_revisit = penalty_revisit

        # Identify materials and exit
        self.materials = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 2]
        assert len(self.materials) == 2, "For simplicity, exactly 2 materials are required."
        self.target_index = {}
        for i, mpos in enumerate(self.materials):
            self.target_index[mpos] = i

        # Exit at bottom-right corner (or define it explicitly)
        self.exit_state = (self.rows-1, self.cols-1)

        # Actions single agent
        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]

        # State dimension: (r1, c1, r2, c2, mask)
        # mask: 2 bits (4 states: 00,01,10,11)
        self.V = np.zeros((self.rows, self.cols, self.rows, self.cols, 4))

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != 1

    def is_terminal(self, r1, c1, r2, c2, mask):
        # Terminal if both materials collected (mask == 3) and both agents on exit
        if (r1,c1) == self.exit_state and (r2,c2) == self.exit_state and mask == 3:
            return True
        return False

    def step(self, r1, c1, r2, c2, mask, a1, a2):
        # Compute next state given a joint action (a1,a2)
        nr1 = r1 + self.actions[a1][0]
        nc1 = c1 + self.actions[a1][1]
        if not self.is_valid(nr1, nc1):
            nr1, nc1 = r1, c1

        nr2 = r2 + self.actions[a2][0]
        nc2 = c2 + self.actions[a2][1]
        if not self.is_valid(nr2, nc2):
            nr2, nc2 = r2, c2

        reward = self.reward_step
        new_mask = mask

        # Check if agent1 collects material
        cell1 = (nr1, nc1)
        if self.grid[nr1,nc1] == 2:
            tidx = self.target_index[cell1]
            if not (mask & (1 << tidx)):
                # collect material
                new_mask = mask | (1 << tidx)
                reward += self.reward_collect

        # Check if agent2 collects material
        cell2 = (nr2, nc2)
        if self.grid[nr2,nc2] == 2:
            tidx2 = self.target_index[cell2]
            if not (new_mask & (1 << tidx2)):
                # collect material
                new_mask = new_mask | (1 << tidx2)
                reward += self.reward_collect

        # If terminal state achieved
        if self.is_terminal(nr1, nc1, nr2, nc2, new_mask):
            reward += 20

        return nr1, nc1, nr2, nc2, new_mask, reward

    def value_iteration(self, max_iterations=1000):
        # Joint action space size = 4 * 4 = 16
        for it in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            for r1 in range(self.rows):
                for c1 in range(self.cols):
                    for r2 in range(self.rows):
                        for c2 in range(self.cols):
                            for mask in range(4):
                                if self.is_terminal(r1,c1,r2,c2,mask) or self.grid[r1,c1] == 1 or self.grid[r2,c2] == 1:
                                    # Terminal or invalid cell states
                                    continue
                                values_actions = []
                                # 16 joint actions
                                for a1 in range(4):
                                    for a2 in range(4):
                                        nr1, nc1, nr2, nc2, nmask, reward = self.step(r1,c1,r2,c2,mask,a1,a2)
                                        values_actions.append(reward + self.gamma * self.V[nr1,nc1,nr2,nc2,nmask])
                                best_value = max(values_actions)
                                delta = max(delta, abs(best_value - self.V[r1,c1,r2,c2,mask]))
                                new_V[r1,c1,r2,c2,mask] = best_value
            self.V = new_V
            if delta < self.theta:
                print(f"Converged after {it} iterations.")
                break

    def extract_policy(self):
        # Extract policy: π(r1,c1,r2,c2,mask) = best joint action
        policy = np.full((self.rows, self.cols, self.rows, self.cols, 4), -1)
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                for r2 in range(self.rows):
                    for c2 in range(self.cols):
                        for mask in range(4):
                            if self.is_terminal(r1,c1,r2,c2,mask) or self.grid[r1,c1]==1 or self.grid[r2,c2]==1:
                                continue
                            values_actions = []
                            for a1 in range(4):
                                for a2 in range(4):
                                    nr1, nc1, nr2, nc2, nmask, reward = self.step(r1,c1,r2,c2,mask,a1,a2)
                                    values_actions.append(reward + self.gamma * self.V[nr1,nc1,nr2,nc2,nmask])
                            best_joint = np.argmax(values_actions)
                            policy[r1,c1,r2,c2,mask] = best_joint
        return policy

    def simulate_policy(self, policy, start_positions, delay=0.5):
        (r1,c1),(r2,c2) = start_positions
        mask = 0
        step_count = 0
        total_reward = 0.0

        while not self.is_terminal(r1,c1,r2,c2,mask):
            print(f"\nÉtape {step_count}:")
            self.print_grid_with_agents(r1,c1,r2,c2,mask)
            joint_action = policy[r1,c1,r2,c2,mask]
            a1 = joint_action //4
            a2 = joint_action %4
            nr1,nc1,nr2,nc2,nmask,reward = self.step(r1,c1,r2,c2,mask,a1,a2)
            total_reward += reward
            r1,c1,r2,c2,mask = nr1,nc1,nr2,nc2,nmask
            step_count += 1

        print(f"\nÉtape {step_count}:")
        self.print_grid_with_agents(r1,c1,r2,c2,mask)
        print(f"Arrivé à l'état terminal en {step_count} étapes.")
        print(f"Récompense totale obtenue: {total_reward}")

    def print_grid_with_agents(self, r1,c1,r2,c2,mask):
        collected = []
        for i in range(2):
            if (mask & (1<<i)) != 0:
                collected.append(i)
        print(f"Matériaux collectés: {collected}")

        for rr in range(self.rows):
            row_str = ""
            for cc in range(self.cols):
                if (rr,cc)==(r1,c1) and (rr,cc)==(r2,c2):
                    row_str += "X "
                elif (rr,cc)==(r1,c1):
                    row_str += "A1"
                elif (rr,cc)==(r2,c2):
                    row_str += "A2"
                else:
                    val = self.grid[rr,cc]
                    if val==1:
                        row_str += "X "
                    elif val==2:
                        tidx = self.target_index[(rr,cc)]
                        if tidx in collected:
                            row_str += ". "
                        else:
                            row_str += "M "
                    else:
                        if (rr,cc) == self.exit_state:
                            row_str += "E "
                        else:
                            row_str += "- "
            print(row_str)

class MultiAgentValueIterationE:
    def __init__(self, grid, gamma=0.9, theta=1e-2, reward_step=-1, reward_clean=10, penalty_revisit=-5, reward_final=50):
        """
        Multi-agent value iteration for E environment:
        - All cells (except obstacles and the exit cell) are dirty at the start.
        - Agents must clean all dirty cells (2), then both go to the exit (0,4).
        
        grid: 5x5 numpy array
            0 = empty/exit
            1 = obstacle
            2 = unclean cell
        Exit = (0,4)
        
        Rewards:
         - step = -1
         - clean a dirty cell first time = +10
         - revisiting a cleaned cell = -5
         - after all cleaned, both at exit = +50
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.gamma = gamma
        self.theta = theta
        self.reward_step = reward_step
        self.reward_clean = reward_clean
        self.penalty_revisit = penalty_revisit
        self.reward_final = reward_final
        
        # Identify all dirty cells
        self.dirty_cells = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 2]
        self.num_dirty = len(self.dirty_cells)
        self.target_index = {}
        for i, dpos in enumerate(self.dirty_cells):
            self.target_index[dpos] = i
        
        # Exit at top-right
        self.exit_state = (0, self.cols-1)
        
        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]  # up, right, down, left
        
        self.V = np.zeros((self.rows, self.cols, self.rows, self.cols, 2**self.num_dirty))
    
    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != 1
    
    def all_cleaned(self, mask):
        return mask == (2**self.num_dirty - 1)
    
    def is_terminal(self, r1, c1, r2, c2, mask):
        # Terminal if all cleaned and both agents at exit
        if self.all_cleaned(mask) and (r1,c1)==self.exit_state and (r2,c2)==self.exit_state:
            return True
        return False
    
    def step(self, r1, c1, r2, c2, mask, a1, a2):
        # next positions
        nr1 = r1 + self.actions[a1][0]
        nc1 = c1 + self.actions[a1][1]
        if not self.is_valid(nr1,nc1):
            nr1, nc1 = r1, c1
        
        nr2 = r2 + self.actions[a2][0]
        nc2 = c2 + self.actions[a2][1]
        if not self.is_valid(nr2,nc2):
            nr2, nc2 = r2, c2
        
        reward = self.reward_step
        new_mask = mask
        
        # Agent 1 cleaning logic
        cell1 = (nr1,nc1)
        if self.grid[nr1,nc1] == 2:
            tidx = self.target_index[cell1]
            already_cleaned = (mask & (1<<tidx))!=0
            if not already_cleaned:
                # first time cleaning
                new_mask = mask | (1<<tidx)
                reward += self.reward_clean
            else:
                # revisit cleaned cell
                reward += self.penalty_revisit
        
        # Agent 2 cleaning logic
        cell2 = (nr2,nc2)
        if self.grid[nr2,nc2] == 2:
            tidx2 = self.target_index[cell2]
            already_cleaned2 = (new_mask & (1<<tidx2))!=0
            if not already_cleaned2:
                new_mask = new_mask | (1<<tidx2)
                reward += self.reward_clean
            else:
                reward += self.penalty_revisit
        
        # Check terminal
        if self.is_terminal(nr1,nc1,nr2,nc2,new_mask):
            reward += self.reward_final
        
        return nr1, nc1, nr2, nc2, new_mask, reward
    
    def value_iteration(self, max_iterations=1000):
        num_joint_actions = 4*4
        for it in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            for r1 in range(self.rows):
                for c1 in range(self.cols):
                    for r2 in range(self.rows):
                        for c2 in range(self.cols):
                            for mask in range(2**self.num_dirty):
                                if self.is_terminal(r1,c1,r2,c2,mask) or self.grid[r1,c1]==1 or self.grid[r2,c2]==1:
                                    continue
                                values_actions = []
                                for a1 in range(4):
                                    for a2 in range(4):
                                        nr1,nc1,nr2,nc2,nmask,reward = self.step(r1,c1,r2,c2,mask,a1,a2)
                                        values_actions.append(reward + self.gamma*self.V[nr1,nc1,nr2,nc2,nmask])
                                best_value = max(values_actions)
                                delta = max(delta, abs(best_value - self.V[r1,c1,r2,c2,mask]))
                                new_V[r1,c1,r2,c2,mask] = best_value
            self.V = new_V
            if delta < self.theta:
                print(f"Converged after {it} iterations.")
                break
    
    def extract_policy(self):
        policy = np.full((self.rows, self.cols, self.rows, self.cols, 2**self.num_dirty), -1)
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                for r2 in range(self.rows):
                    for c2 in range(self.cols):
                        for mask in range(2**self.num_dirty):
                            if self.is_terminal(r1,c1,r2,c2,mask) or self.grid[r1,c1]==1 or self.grid[r2,c2]==1:
                                continue
                            values_actions = []
                            for a1 in range(4):
                                for a2 in range(4):
                                    nr1,nc1,nr2,nc2,nmask,reward = self.step(r1,c1,r2,c2,mask,a1,a2)
                                    values_actions.append(reward + self.gamma*self.V[nr1,nc1,nr2,nc2,nmask])
                            best_joint = np.argmax(values_actions)
                            policy[r1,c1,r2,c2,mask] = best_joint
        return policy
    
    def simulate_policy(self, policy, start_positions, delay=0.5):
        (r1,c1),(r2,c2) = start_positions
        mask = 0
        step_count = 0
        total_reward = 0.0
        
        while not self.is_terminal(r1,c1,r2,c2,mask):
            print(f"\nÉtape {step_count}:")
            self.print_grid_with_agents(r1,c1,r2,c2,mask)
            joint_action = policy[r1,c1,r2,c2,mask]
            a1 = joint_action //4
            a2 = joint_action %4
            nr1,nc1,nr2,nc2,nmask,reward = self.step(r1,c1,r2,c2,mask,a1,a2)
            total_reward += reward
            r1,c1,r2,c2,mask = nr1,nc1,nr2,nc2,nmask
            step_count += 1
        
        print(f"\nÉtape {step_count}:")
        self.print_grid_with_agents(r1,c1,r2,c2,mask)
        print(f"Arrivé à l'état terminal en {step_count} étapes.")
        print(f"Récompense totale obtenue: {total_reward}")

    def print_grid_with_agents(self, r1,c1,r2,c2,mask):
        cleaned_list = []
        for i in range(self.num_dirty):
            if (mask & (1<<i)) != 0:
                cleaned_list.append(i)
        print(f"Cellules nettoyées: {cleaned_list}")

        for rr in range(self.rows):
            row_str = ""
            for cc in range(self.cols):
                if (rr,cc)==(r1,c1) and (rr,cc)==(r2,c2):
                    row_str += "X "
                elif (rr,cc)==(r1,c1):
                    row_str += "A1"
                elif (rr,cc)==(r2,c2):
                    row_str += "A2"
                else:
                    val = self.grid[rr,cc]
                    if val == 1:
                        row_str += "X "
                    elif val == 2:
                        tidx = self.target_index[(rr,cc)]
                        if tidx in cleaned_list:
                            row_str += "C "
                        else:
                            row_str += "U "
                    else:
                        if (rr,cc) == self.exit_state:
                            row_str += "E "
                        else:
                            row_str += ". "
            print(row_str)



class MultiAgentValueIterationG:
    def __init__(self, grid, gamma=0.9, theta=1e-4, reward_step=-1, reward_station=100):
        """
        Multi-agent Value Iteration for G environment:
        - Agents start in the bottom-left column.
        - Two stations (3) in the grid.
        - Each agent must reach a distinct station simultaneously.

        grid:
            0 = empty
            1 = obstacle
            3 = station

        We assume exactly 2 stations for simplicity.
        
        When both agents are on distinct stations at the same time, the episode ends (terminal state) and a large reward is given.

        Rewards:
          - step = -1
          - If an agent reaches a station and the other is also on a different station, terminal and +100 reward given.
          - If only one agent is on a station, no immediate large reward until the second agent also reaches the other station.
          - Agents must be on distinct stations (not the same one).
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.gamma = gamma
        self.theta = theta
        self.reward_step = reward_step
        self.reward_station = reward_station

        # Identify stations
        self.stations = [(r,c) for r in range(self.rows) for c in range(self.cols) if self.grid[r,c] == 3]
        assert len(self.stations) == 2, "For simplicity, exactly 2 stations are required."

        # No mask needed, the terminal condition is both agents on different stations
        # State: (r1,c1,r2,c2)

        self.actions = [(-1,0),(0,1),(1,0),(0,-1)]  # up,right,down,left

        # Dimension of V: rows x cols x rows x cols (no mask)
        self.V = np.zeros((self.rows, self.cols, self.rows, self.cols))

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r,c] != 1

    def is_terminal(self, r1, c1, r2, c2):
        # Terminal if both agents on distinct stations
        # Check if r1,c1 is a station and r2,c2 is a station, and not the same one
        if (r1,c1) in self.stations and (r2,c2) in self.stations and (r1,c1) != (r2,c2):
            return True
        return False

    def step(self, r1, c1, r2, c2, a1, a2):
        nr1 = r1 + self.actions[a1][0]
        nc1 = c1 + self.actions[a1][1]
        if not self.is_valid(nr1, nc1):
            nr1, nc1 = r1, c1

        nr2 = r2 + self.actions[a2][0]
        nc2 = c2 + self.actions[a2][1]
        if not self.is_valid(nr2, nc2):
            nr2, nc2 = r2, c2

        reward = self.reward_step

        # Check if terminal:
        if self.is_terminal(nr1,nc1,nr2,nc2):
            reward += self.reward_station

        return nr1, nc1, nr2, nc2, reward

    def value_iteration(self, max_iterations=1000):
        for it in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            for r1 in range(self.rows):
                for c1 in range(self.cols):
                    for r2 in range(self.rows):
                        for c2 in range(self.cols):
                            if self.is_terminal(r1,c1,r2,c2) or self.grid[r1,c1]==1 or self.grid[r2,c2]==1:
                                # terminal or invalid states
                                continue
                            values_actions = []
                            for a1 in range(4):
                                for a2 in range(4):
                                    nr1,nc1,nr2,nc2,reward = self.step(r1,c1,r2,c2,a1,a2)
                                    values_actions.append(reward + self.gamma*self.V[nr1,nc1,nr2,nc2])
                            best_value = max(values_actions)
                            delta = max(delta, abs(best_value - self.V[r1,c1,r2,c2]))
                            new_V[r1,c1,r2,c2] = best_value
            self.V = new_V
            if delta < self.theta:
                print(f"Converged after {it} iterations.")
                break

    def extract_policy(self):
        policy = np.full((self.rows, self.cols, self.rows, self.cols), -1)
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                for r2 in range(self.rows):
                    for c2 in range(self.cols):
                        if self.is_terminal(r1,c1,r2,c2) or self.grid[r1,c1]==1 or self.grid[r2,c2]==1:
                            continue
                        values_actions = []
                        for a1 in range(4):
                            for a2 in range(4):
                                nr1,nc1,nr2,nc2,reward = self.step(r1,c1,r2,c2,a1,a2)
                                values_actions.append(reward + self.gamma*self.V[nr1,nc1,nr2,nc2])
                        best_joint = np.argmax(values_actions)
                        policy[r1,c1,r2,c2] = best_joint
        return policy

    def simulate_policy(self, policy, start_positions, delay=0.5):
        (r1,c1),(r2,c2) = start_positions
        step_count = 0
        total_reward = 0.0

        while not self.is_terminal(r1,c1,r2,c2):
            print(f"\nÉtape {step_count}:")
            self.print_grid_with_agents(r1,c1,r2,c2)
            joint_action = policy[r1,c1,r2,c2]
            a1 = joint_action //4
            a2 = joint_action %4
            nr1,nc1,nr2,nc2,reward = self.step(r1,c1,r2,c2,a1,a2)
            total_reward += reward
            r1,c1,r2,c2 = nr1,nc1,nr2,nc2
            step_count += 1

        print(f"\nÉtape {step_count}:")
        self.print_grid_with_agents(r1,c1,r2,c2)
        print(f"Arrivé à l'état terminal en {step_count} étapes.")
        print(f"Récompense totale obtenue: {total_reward}")

    def print_grid_with_agents(self, r1,c1,r2,c2):
        for rr in range(self.rows):
            row_str = ""
            for cc in range(self.cols):
                if (rr,cc)==(r1,c1) and (rr,cc)==(r2,c2):
                    row_str += "X "
                elif (rr,cc)==(r1,c1):
                    row_str += "A1"
                elif (rr,cc)==(r2,c2):
                    row_str += "A2"
                else:
                    val = self.grid[rr,cc]
                    if val==1:
                        row_str += "X "
                    elif val==3:
                        # Station
                        row_str += "S "
                    else:
                        row_str += ". "
            print(row_str)





if __name__ == "__main__":
    # Example usage
    grid = np.array([
        [0, 0, 0, 0],
        [0, 0, 2, 1],
        [2, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    start_positions=((0,0),(0,1))
    vi_ma = MultiAgentValueIteration(grid)
    vi_ma.value_iteration(max_iterations=50)
    policy = vi_ma.extract_policy()
    vi_ma.simulate_policy(policy, start_positions)


    E = np.array([
        [0, 0, 2],
        [2, 2, 2],
        [2, 2, 2]
    ])
    # Dirty cells: (0,1), (0,2), (1,0) => 3 dirty cells => mask has size 2^3=8 states
    # Exit at top-right = (0,2)
    # Start positions for agents: must be empty cells
    start_positions = ((0,0),(0,1)) # just an example, ensure they are not obstacles or dirty
    vi_ma_e = MultiAgentValueIterationE(E, gamma=0.9, theta=1e-2, reward_step=-1, reward_clean=10, penalty_revisit=-5, reward_final=50)
    vi_ma_e.value_iteration(max_iterations=50)
    policy_e = vi_ma_e.extract_policy()
    vi_ma_e.simulate_policy(policy_e, start_positions)



    # 3 = stations, we must have exactly 2 stations
    G = np.array([
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 0]
    ])
    # Two stations at (0,2) and (3,2)
    # Agents start at bottom-left column: let's place them at (3,0) and (3,0) or (3,0) and (3,1)
    start_positions = ((3,0),(3,0)) # both agents start in bottom-left corner (same cell)
    vi_ma_g = MultiAgentValueIterationG(G, gamma=0.9, theta=1e-4, reward_step=-1, reward_station=100)
    vi_ma_g.value_iteration(max_iterations=100)
    policy_g = vi_ma_g.extract_policy()
    vi_ma_g.simulate_policy(policy_g, start_positions)