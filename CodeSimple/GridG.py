
import random
import heapq


class GridG:
    def __init__(self, size, num_agents, num_obstacles):
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

        # Place recharge stations, one per agent
        for _ in range(num_agents):
            self._place_item('R', self.stations)

    def _place_item(self, symbol, collection):
        """Place an item randomly in the grid."""
        while True:
            x, y = random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1)
            if not self.grid[x][y]:
                self.grid[x][y] = symbol
                if collection is not None:
                    collection.append((x, y))
                if symbol == 'O':  # Mark obstacles as bad states
                    self.bad_states.add((x, y))
                return (x, y)


    def display(self):
        """Display the grid."""
        print("\nGrid G:")
        for row in self.grid:
            print(' '.join(cell if cell else '.' for cell in row))

class AgentG:
    def __init__(self, initial_position, grid, agent_id):
        self.position = initial_position
        self.grid = grid
        self.target = None
        self.recharged = False
        self.agent_id = agent_id  # Assign a unique ID

    def set_target(self, target):
        """Set the agent's target (station)."""
        self.target = target

    def move(self):
        if self.recharged:
            return

        # Find the nearest station if no target is assigned
        if self.target is None:
            self.target = self.find_nearest_station()

        # Reassign target if the current one is invalid or already used
        if self.target not in self.grid.stations:
            print(f"Agent {self.agent_id} lost target {self.target}.")
            self.recharged = True
            return

        path = self.a_star_search(self.position, self.target)
        if path:
            new_position = path[0]
            # Update position
            self.grid.grid[self.position[0]][self.position[1]] = ''
            self.position = new_position
            self.grid.grid[new_position[0]][new_position[1]] = self.agent_id

            # Check if the agent reached the station
            if self.position == self.target:
                self.recharged = True
                if self.target in self.grid.stations:  # Double-check station is still valid
                    self.grid.stations.remove(self.target)
                print(f"Agent {self.agent_id} recharged at station {self.target}!")

    def find_nearest_station(self):
        """Find the nearest station using Manhattan distance."""
        nearest_station = None
        min_distance = float('inf')

        for station in self.grid.stations:
            # Manhattan distance
            distance = abs(self.position[0] - station[0]) + abs(self.position[1] - station[1])
            if distance < min_distance:
                min_distance = distance
                nearest_station = station

        return nearest_station

    def heuristic(self, a, b):
        """Heuristic function for A* search (Manhattan distance)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_search(self, start, goal):
        """A* search algorithm to find a path from start to goal."""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

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

        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def neighbors(self, node):
        """Get valid neighbors of a node."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = node[0] + dx, node[1] + dy
            if (x, y) in self.grid.states and (x, y) not in self.grid.bad_states:
                neighbors.append((x, y))
        return neighbors

    def __repr__(self):
        return f"AgentG(position={self.position}, target={self.target})"