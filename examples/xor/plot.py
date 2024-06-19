import matplotlib.pyplot as plt
import csv

x = []
y = []
z = []

with open('examples/xor/xor-boundry.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            pass
        else:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
        line_count += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c=z)
plt.show()
