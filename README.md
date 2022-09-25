# con-pol-opt
This repository is the code for the case study in "Constrained Policy Optimization for Stochastic Optimal Control under" by Sungho Shin, Francois Pacaud, and Mihai Anitescu.

## How to run

First, install [Julia](https://julialang.org/downloads/).

Clone this repository
```
git clone https://github.com/sshin23/con-pol-opt.git
```

The dependencies can be installed by running
```
cd con-pol-opt
julia --project -e "import Pkg; Pkg.instantiate()"
```
Note: to run the code with efficient HSL solvers (necessary for MPC simulation), consider setting the environment variables for [MadNLPHSL](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPHSL) before instantiating the project.

Finally, run the code
```
julia --project example.jl
```
