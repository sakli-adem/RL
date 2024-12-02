import random
import heapq

class GridE:
    def __init__(self, size, num_obstacles):
        self.size = size
        self.grid = [['U' for _ in range(size[1])] for _ in range(size[0])]
        self.agents = []
        self.unclean_cells = [(x, y) for x in range(size[0]) for y in range(size[1])]
        self.states = [(x, y) for x in range(size[0]) for y in range(size[1])]
        self.bad_states = set()
        self.obstacles = []

        # Place obstacles
        for _ in range(num_obstacles):
            self._place_item('O', self.obstacles)

    def _place_item(self, symbol, collection):
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
            if self.grid[x][y] == 'U':
                self.grid[x][y] = symbol
                if collection is not None:
                    collection.append((x, y))
                if symbol == 'O':  # Mark obstacles as bad states
                    self.bad_states.add((x, y))
                    if (x, y) in self.unclean_cells:
                        self.unclean_cells.remove((x, y))
                return (x, y)
            attempts += 1

        raise ValueError(f"Could not place {symbol} after {max_attempts} attempts")

    def display(self):
        print("\nGrid E (Warehouse):")
        for row in self.grid:
            print(' '.join(cell if cell else '.' for cell in row))
        print(f"Unclean cells remaining: {len(self.unclean_cells)}")

    def is_cleaning_complete(self):
        return len(self.unclean_cells) == 0

class AgentE:
    def __init__(self, initial_position, grid, agent_id):
        self.position = initial_position
        self.grid = grid
        self.agent_id = agent_id
        self.target = None
        self.just_entered = True  # Flag to indicate the agent has just entered Grid E

    def move(self):
        if self.just_entered:
            self.just_entered = False  # Reset the flag after the first move
            return  # Do not move during the first iteration

        # Find the nearest unclean cell if no target is assigned
        if self.target is None or self.target not in self.grid.unclean_cells:
            self.target = self.find_nearest_unclean_cell()

        # If no unclean cells remain, do nothing
        if self.target is None:
            print(f"Agent {self.agent_id}: No unclean cells left")
            return

        # Debug: Print agent's current position and target
        print(f"Agent {self.agent_id}: Current pos {self.position}, Target {self.target}")

        # If target is the current position, find a different target
        if self.target == self.position:
            print(f"Agent {self.agent_id}: Target is current position, finding new target")
            # Find a different unclean cell
            alternative_targets = [
                cell for cell in self.grid.unclean_cells
                if cell != self.position and cell not in self.grid.bad_states
            ]

            if not alternative_targets:
                print(f"Agent {self.agent_id}: No alternative targets found")
                return

            # Choose the closest alternative target
            self.target = min(
                alternative_targets,
                key=lambda cell: (
                    abs(self.position[0] - cell[0]) + abs(self.position[1] - cell[1])
                )
            )

        # Find path to the target
        path = self.a_star_search(self.position, self.target)

        if not path:
            print(f"Agent {self.agent_id}: No path found to {self.target}")
            return

        # Move to the next cell in the path
        new_position = path[0]

        # Update grid
        if self.grid.grid[self.position[0]][self.position[1]] == self.agent_id:
            self.grid.grid[self.position[0]][self.position[1]] = 'C'  # Mark as cleaned if agent was there
        self.position = new_position
        self.grid.grid[new_position[0]][new_position[1]] = self.agent_id

        # Check if the agent reached the unclean cell
        if self.position == self.target:
            if self.target in self.grid.unclean_cells:
                self.grid.unclean_cells.remove(self.target)
                print(f"Agent {self.agent_id} cleaned cell at {self.target}!")
                self.target = None  # Reset target after cleaning


    def find_nearest_unclean_cell(self):
        """
        Find the nearest unclean cell using Manhattan distance.
        """
        if not self.grid.unclean_cells:
            return None

        # Filter out obstacle cells and the current position
        valid_unclean_cells = [
            cell for cell in self.grid.unclean_cells
            if cell not in self.grid.bad_states and cell != self.position
        ]

        if not valid_unclean_cells:
            return None

        return min(
            valid_unclean_cells,
            key=lambda cell: (
                abs(self.position[0] - cell[0]) + abs(self.position[1] - cell[1])
            )
        )

    def a_star_search(self, start, goal):
        """
        A* search algorithm to find a path from start to goal.
        """
        if start == goal:
            return []

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        # Reconstruct path
        path = []
        current = goal
        while current != start:
            if current not in came_from:
                print(f"No path found from {start} to {goal}")
                return []  # No path found
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, node):
        """Get valid neighboring cells."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = node[0] + dx, node[1] + dy
            if (x, y) in self.grid.states and (x, y) not in self.grid.bad_states:
                neighbors.append((x, y))
        return neighbors