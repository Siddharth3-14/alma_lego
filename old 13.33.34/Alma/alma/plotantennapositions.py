#! /bin/env python3
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))

with open('ant_pos.txt') as f:
    for line in f:
        if line.startswith('ant'): 
            continue
        line = line.split()    
        #line[2] = - float(line[2])  # reversed for plotting the back-side layout
        plt.plot(float(line[1]), float(line[2]),'o')     
        plt.text(float(line[1]), float(line[2]), line[0])     

plt.savefig('plotantennapositions.png')        
plt.show()        
