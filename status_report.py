import os

print(len(os.listdir("hp_optimization/train_17/trail_30/threads")), "/", 90)
with open("job2_outs.txt", "r") as f:
    print(f.read())
