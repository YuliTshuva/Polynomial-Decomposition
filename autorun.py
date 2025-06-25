import subprocess

n = 10

# Run train_18.py n times
for i in range(n):
    subprocess.run(["python", "train_16.py"])
