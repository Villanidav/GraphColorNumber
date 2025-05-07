#!/bin/bash
#SBATCH --nodes=8
#SBATVH --ntasks=16             # ntasks = 2 * nodes
#SBATCH --ntasks-per-node=2     # 2 CPUs per node on VEGA
#SBATCH --time=02:46:40
#SBATCH --output %j.output
#SBATCH --partition=cpu

srun --mpi=pmix_v3 ./my_program

