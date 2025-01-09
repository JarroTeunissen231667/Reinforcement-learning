# Robotic Pipette Simulation - Working Envelope Determination
This repository provides a simulation environment for the Opentron OT-2 robotic system, focusing on determining the working envelope of the pipette's tip. The simulation demonstrates the ability to send commands to the robot, receive observations about its state, and calculate the working envelope by exploring the extremities of the robot's movement.
---
## Environment Setup
Follow these steps to set up and run the simulation:
### 1. Install Python
Ensure you have Python 3.8 or later installed. Download it from [python.org](https://www.python.org).
### 2. Create a Virtual Environment
Create a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
Install the required libraries using:
```bash
pip install -r requirements.txt
pip install pybullet
```
#### List of Dependencies:
- `numpy` - For numerical calculations
- `matplotlib` - For optional visualizations
- `simpy` - For simulation handling
- Any other custom libraries specific to the Opentron OT-2 simulator (ensure they are added to `requirements.txt`).
### 4. Interface with the Simulation Environment
To understand how to interface with the Opentron OT-2 robotic simulation, refer to the [Simulation Environment Setup Guide](https://example-link-to-simulation-guide).
### 5. Run the Simulation
Execute the main script to determine the working envelope:
```bash
corners.ipynb
```
---
## Working Envelope of the Pipette
The pipette's working envelope represents the cube formed by the furthest points the pipette can reach. By adjusting motor velocities for each axis, the simulation moves the pipette to the corners of this cube, recording the coordinates at each extreme.
### Determined Working Envelope
- **X-Axis Range**: `-0.18700` to `0.25300`
- **Y-Axis Range**: `-0.17050` to `0.21950`
- **Z-Axis Range**: `0.16940` to `0.28950`
### Recorded Corner Points:
1. `[-0.18700, 0.21950, 0.16950]`
2. `[0.25300, -0.17050, 0.16940]`
3. `[0.25300, -0.17050, 0.28950]`
4. `[-0.18700, 0.21950, 0.28950]`
5. `[-0.18700, -0.17050, 0.16950]`
6. `[0.25300, 0.21950, 0.16950]`
7. `[0.25300, 0.21950, 0.28950]`
8. `[-0.18700, -0.17050, 0.28950]`
---
## Deliverables
1. **Code**: Well-documented Python code for running the simulation:
   - Sends commands to the robot.
   - Retrieves observations of the robot's state.
   - Determines and outputs the pipette's working envelope.
2. **README**: This document provides:
   - Environment setup instructions.
   - List of dependencies.
   - Description of the pipette's working envelope.
3. **Optional Deliverable**:
   - A GIF demonstrating the robot moving to the envelope's corners and printing its observations. (Generate using a screen recording tool and save as `robot_simulation.gif` in the repository).
---
## License
This project is licensed under the MIT License.