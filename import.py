from sim_class import Simulation
sim = Simulation(num_agents=1)

# Define the velocities to move to all 8 corners
corner_velocities = [
    [-0.1, 0.1, -0.1, 0],  # Top-left
    [0.1, 0.1, -0.1, 0],   # Top-right
    [0.1, -0.1, -0.1, 0],  # Bottom-right
    [-0.1, -0.1, -0.1, 0], # Bottom-left
    [-0.1, 0.1, 0.1, 0],   # Top-left (up)
    [0.1, 0.1, 0.1, 0],    # Top-right (up)
    [0.1, -0.1, 0.1, 0],   # Bottom-right (up)
    [-0.1, -0.1, 0.1, 0]   # Bottom-left (up)
]
 
# Run the simulation for each set of velocities and print the maximum pipette position
for velocities in corner_velocities:
    actions = [velocities]
    max_pipette_position = [float('-inf'), float('-inf'), float('-inf')]
    for _ in range(1000):
        state = sim.run(actions)
        pipette_position = state['robotId_1']['pipette_position']
        max_pipette_position = [max(max_pipette_position[i], pipette_position[i]) for i in range(3)]
    print(max_pipette_position)