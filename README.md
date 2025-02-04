# Finite-Element-Simulation
Finite Element Solver for 2D State Heat Equation and Stress Equilibrium Equation for Plane Stress of Plane Strain Problems
## Installation
To run the code, you need to have Python installed. Additionally, you'll need the following libraries:
* '```numpy```'
* '```matplotlib```'
* '```tqdm```'
* '```sympy```'

  You can install these libraries using pip:
  ``` pip install numpy matplotlib tqdm sympy```

## Usage
The code for meshing, and calculation fo the elemental and global stiffness matrix is contained in the python files `advait_solver.py` and `advait_solver_solid_mech.py`.
The simulation parameters are to be adjusted by calling the script, Implementation of boundary conditions has to be done manually (the code does require some cleanup), please go through the sample use cases encapped in examples.
