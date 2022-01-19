import os
dir_predicted = '/home/ellentuane/Documents/IC/detected/960/1'
lines = []

for filename in os.listdir(dir_predicted):
    # DETECTED FILE NAME
    for k in filename:
        img_name = filename.split("_")

    with open(os.path.join(dir_predicted, filename), "r") as files:
        for line in files:
            lines.append(line)

print(filename)
print(img_name)