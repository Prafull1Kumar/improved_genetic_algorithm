# import numpy as np
# import matplotlib.pyplot as plt

# csa = [95, 92, 87, 85,97,93,67]
# ga = [97, 60, 20, 76,46,18,0]

# n=7
# r = np.arange(n)
# width = 0.25


# plt.bar(r, csa, color = 'b',
# 		width = width, edgecolor = 'black',
# 		label='Crow Search Algorithm')
# plt.bar(r + width, ga, color = 'g',
# 		width = width, edgecolor = 'black',
# 		label='Genetic Algorithm')

# plt.xlabel("Expression")
# plt.ylabel("Solution found")
# plt.title(" Accurracy of the algorithm our CSA and GA ")

# # plt.grid(linestyle='--')
# plt.xticks(r + width/2,['x','4*x*x+7*x+13','x*x*x*x+4*x*x+7*x+11'
#                         ,'3*x*x*x + 4*y*y','12*x*x*x*y*y','7*x*x*x+5*y*y+11*z','3*x*x*x +4*y*y*x+11*y*y'])

# plt.legend()

# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data for the bars
categories = ['x','4*x*x+7*x+13','x*x*x*x+4*x*x+7*x+11'
                        ,'3*x*x*x + 4*y*y','12*x*x*x*y*y','7*x*x*x+5*y*y+11*z','3*x*x*x +4*y*y*x+11*y*y']
csa = [100, 92, 87, 85,97,93,70]
dev = [0, 0, 0, 0,0,0,0]
ga = [100, 60, 20, 76,46,18,10]
csa_avg_gen =['1.0 gen','22.5 gen','34.8 gen','17.7 gen','13.5 gen','45.0 gen','60.7 gen']
ga_avg_gen=['1.2 gen','41.4 gen','32.0 gen','74.8 gen','44.0 gen','55.0 gen','30 gen']
csa_avg_dev =[0,0.0,21.0,12117.0,0.0,0.0,102521.0]
ga_avg_dev=[0,596.0,50250.0,118442.0,59138436.0,26369710.0,1201428.0]
# Set the width of the bars
bar_width = 0.35

# Calculate the x positions for the bars
x = np.arange(len(categories))

# Create the bar plots
plt.bar(x, csa, width=bar_width, label='CSBGA')
plt.bar(x + bar_width, ga, width=bar_width, label='GA')
plt.bar(x + 2*bar_width, dev,color = 'yellow', width=bar_width, label='Deviation')

# Add labels inside the bars
for i, (gen,value) in enumerate(zip(csa_avg_gen,csa)):
    plt.text(i, value+2, str(gen), ha='center', va='center')
for i, (gen,value) in enumerate(zip(ga_avg_gen,ga)):
    plt.text(i + bar_width, value+2, str(gen), ha='center', va='center')

# Add labels above the bars
for i, (dev,value) in enumerate(zip(csa_avg_dev,csa)):
    plt.text(i, value/2, str(dev), ha='center', va='bottom',color='yellow')
for i, (gen,value) in enumerate(zip(ga_avg_dev,ga)):
    plt.text(i + bar_width, value/2, str(dev), ha='center', va='bottom',color='yellow')

# Add labels and title
plt.xlabel('Expressions')
plt.ylabel('Solution found')
plt.title(' Accurracy of the algorithm our CSBGA and GA ')
plt.xticks(x + 2*bar_width / 3, categories)
plt.legend()

# Display the plot
plt.show()
