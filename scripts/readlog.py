with open("logs/tdfann.log", "r") as f:
    lines = f.readlines()
    print(lines[-2].split(' ')[-2], lines[-1].split(' ')[-1])