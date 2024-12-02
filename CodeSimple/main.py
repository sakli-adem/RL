from GridH import GridH, AgentH
from GridE import GridE, AgentE
from GridG import GridG, AgentG

def display_all_grids(grid_h, grid_e=None, grid_g=None):
    """Display all active grids with file names."""
    print("\n" + "="*50)
    print("CURRENT GRID STATE")
    print("="*50)

    # Display GridH with its file name
    print("\nGrid H (from GridH.py):")
    for row in grid_h.grid:
        print(' '.join(cell if cell else '.' for cell in row))

    # Display GridE with its file name if it exists
    if grid_e:
        print("\nGrid E (Warehouse, from GridE.py):")
        for row in grid_e.grid:
            print(' '.join(cell if cell else '.' for cell in row))
        print(f"Unclean cells remaining: {len(grid_e.unclean_cells)}")

    # Display GridG with its file name if it exists
    if grid_g:
        print("\nGrid G (from GridG.py):")
        for row in grid_g.grid:
            print(' '.join(cell if cell else '.' for cell in row))

def move_agent_to_grid_e(grid_h, grid_e, agent_h):
    """Move a specific agent from GridH to GridE."""
    entry_position = (0, 0)  # Always start at (0, 0)
    grid_e.grid[entry_position[0]][entry_position[1]] = agent_h.agent_id
    agent_e = AgentE(entry_position, grid_e, agent_id=agent_h.agent_id)
    agent_e.just_entered = True  # Set the just_entered flag
    grid_e.agents.append(agent_e)

    # Remove agent from GridH
    grid_h.grid[agent_h.position[0]][agent_h.position[1]] = ''

    return agent_e

if __name__ == "__main__":
    # Get user input for grid size, number of agents, materials, and obstacles
    size_x = int(input("Insert the x (horizontal) size of the grid: "))
    size_y = int(input("Insert the y (vertical) size of the grid: "))
    num_agents = int(input("Insert the number of agents: "))
    num_materials = int(input("Insert the number of materials: "))
    num_obstacles = int(input("Insert the number of obstacles: "))

    # Create the initial grid H
    grid_h = GridH(size=(size_x, size_y), num_agents=num_agents, num_materials=num_materials, num_obstacles=num_obstacles)
    display_all_grids(grid_h)
    input("Press Enter to continue...")

    # Stage 1: Collecting materials in Grid H
    while any(not agent.collected for agent in grid_h.agents) or grid_h.materials:
        for agent in grid_h.agents:
            agent.move()
        display_all_grids(grid_h)
        input("Press Enter to continue...")

    # Create GridE
    grid_e = GridE(size=(size_x, size_y), num_obstacles=num_obstacles)

    # Stage 2: Moving all agents to bottom-right and entering GridE
    grid_e_agents = []
    remaining_agents = grid_h.agents.copy()

    while remaining_agents:
        for agent in remaining_agents[:]:  # Create a copy of the list to modify during iteration
            if agent.position != (grid_h.size[0] - 1, grid_h.size[1] - 1):
                # Move to bottom-right corner
                goal_position = (grid_h.size[0] - 1, grid_h.size[1] - 1)
                path = agent.a_star_search(agent.position, goal_position)
                if path:
                    new_position = path[0]
                    agent.grid.grid[agent.position[0]][agent.position[1]] = ''
                    agent.position = new_position
                    agent.grid.grid[new_position[0]][new_position[1]] = agent.agent_id
            else:
                # Agent has reached bottom right, move to GridE
                agent_e = move_agent_to_grid_e(grid_h, grid_e, agent)
                grid_e_agents.append(agent_e)
                remaining_agents.remove(agent)

        # Move agents in GridE
        for agent_e in grid_e.agents:
            agent_e.move()

        display_all_grids(grid_h, grid_e)
        input("Press Enter to continue...")

    # Stage 3: Cleaning in GridE
    while not grid_e.is_cleaning_complete():
        for agent in grid_e.agents:
            agent.move()
        display_all_grids(grid_h, grid_e)
        input("Press Enter to continue...")

    # Ensure agents reach top-right column of GridE simultaneously
    goal_positions = [(i, grid_e.size[1] - 1) for i in range(len(grid_e.agents))]
    while any(agent_e.position != goal_positions[i] for i, agent_e in enumerate(grid_e.agents)):
        for i, agent_e in enumerate(grid_e.agents):
            goal_position = goal_positions[i]
            path = agent_e.a_star_search(agent_e.position, goal_position)
            if path:
                new_position = path[0]
                agent_e.grid.grid[agent_e.position[0]][agent_e.position[1]] = 'C'
                agent_e.position = new_position
                agent_e.grid.grid[new_position[0]][new_position[1]] = agent_e.agent_id

        display_all_grids(grid_h, grid_e)
        input("Press Enter to continue...")

    # Stage 4: Moving to GridG
    grid_g = GridG(size=(size_x, size_y), num_agents=num_agents, num_obstacles=num_obstacles)

    # Move agents from GridE to GridG, one at a time
    for agent_e in grid_e.agents[:]:
        entry_position = (grid_g.size[0] - 1, 0)  # Bottom-left corner
        grid_g.grid[entry_position[0]][entry_position[1]] = agent_e.agent_id
        agent_g = AgentG(entry_position, grid_g, agent_id=agent_e.agent_id)
        grid_g.agents.append(agent_g)

        # Remove agent from GridE
        grid_e.grid[agent_e.position[0]][agent_e.position[1]] = 'C'  # Mark the old position as cleaned
        grid_e.agents.remove(agent_e)  # Remove from Grid E's agent list

        display_all_grids(grid_h, grid_e, grid_g)
        input(f"Agent {agent_e.agent_id} transitioned to Grid G. Press Enter to continue...")

    # Assign unique stations to agents at the start
    for i, agent in enumerate(grid_g.agents):
        if i < len(grid_g.stations):
            agent.set_target(grid_g.stations[i])  # Assign each agent a unique station
        else:
            print(f"Agent at {agent.position} has no station assigned yet.")

    # Stage 5: Recharging in GridG
    while any(not agent.recharged for agent in grid_g.agents):
        for agent in grid_g.agents:
            agent.move()
        display_all_grids(grid_h, grid_e, grid_g)
        input("Press Enter to continue...")

    print("All agents have recharged!")
    display_all_grids(grid_h, grid_e, grid_g)
