#!/bin/bash
#SBATCH --job-name=job1  # job name
#SBATCH --output=job1.out        # stdout goes here
#SBATCH --error=job1.err         # errors go here
#SBATCH --partition=long         # can be short, long, or gpu
#SBATCH --cpus-per-task=8        # number of CPU cores
#SBATCH --mem=32G                # memory (RAM)
#SBATCH --time=100:00:00          # max runtime

# --- Your code below ---
cd /cs/labs/dshahaf/yuli.tshuva/projects/polynomial_decomposition
source venv/bin/activate
python3 pipeline.py
