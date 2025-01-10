# Task 9: Simulation Environment

## Environment Setup
To establish the simulation environment, the repository from GitHub was cloned using the following link: [Y2B-2023-OT2_Twin Repository](https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git). Additionally, the Visual Studio C++ Toolkit was installed to facilitate the integration and functionality of **pybullet**. A script named `task_9.py` was then developed to initiate and run the simulation effectively.

## Prerequisites
The project requires the installation of specific packages and access to certain files. Ensure the following dependencies are met:
- **pybullet** version `3.2.6`
- **numpy** version `1.26.4`
- Files from the Y2B-2023-OT2_Twin repository: [Y2B-2023-OT2_Twin Repository](https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git)

## Determining the Operational Envelope
To identify the operational envelope, various velocity combinations (0 and 1) were applied. By monitoring and logging the coordinates of the pippet at each timestep, the positions of each corner were sequentially recorded. These coordinates were then incorporated into the agent's navigation path. Subsequently, a loop was implemented to guide the agent through these corner points, ensuring accurate visualization within the envelope. The coordinates defining the operational envelope are as follows:

1. **Right Front Bottom**: `[-0.18700, 0.21950, 0.16950]`
2. **Right Front Top**: `[0.25300, -0.17050, 0.16940]`
3. **Left Front Top**: `[0.25300, -0.17050, 0.28950]`
4. **Left Front Bottom**: `[-0.18700, 0.21950, 0.28950]`
5. **Left Back Bottom**: `[-0.18700, -0.17050, 0.16950]`
6. **Left Back Top**: `[0.25300, 0.21950, 0.16950]`
7. **Right Back Top**: `[0.25300, 0.21950, 0.28950]`
8. **Right Back Bottom**: `[-0.18700, -0.17050, 0.28950]`

These coordinates outline the boundaries within which the agent operates, ensuring consistent and accurate movement within the simulated environment.
