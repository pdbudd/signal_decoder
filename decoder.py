import numpy as np
import matplotlib.pyplot as plt
import math

complex_numbers = np.fromfile("AAAAAAAAA.dat", dtype=np.complex64)

absolute = []
Dphi=[]
NRZI_decoded=[]

phi_priori = 0
print(len(complex_numbers))
for i in complex_numbers:
    if abs(i) > 0.1:
        absolute.append(abs(i))
        phi = math.atan2(i.imag, i.real)
        # Phase change calculation
        delta_phi = phi - phi_priori
        # Adjust phase change to avoid jumps
        while delta_phi > math.pi:
            delta_phi -= 2 * math.pi
        while delta_phi < -math.pi:
            delta_phi += 2 * math.pi
        Dphi.append(delta_phi)
        phi_priori = phi

for i in range(len(Dphi) - 1):
    if (Dphi[i + 1] < 0.05 and Dphi[i] > 0.05): #or (Dphi[i + 1] > 0.05 and Dphi[i] < 0.05):
        NRZI_decoded.append(1)
    else:
        NRZI_decoded.append(0)

print(len(absolute))

# Define the range you want to plot, for example, the first 100 elements
start = 0
end = len(Dphi)

# Plotting
plt.figure(figsize=(10, 5))

# Plot real parts
plt.plot(NRZI_decoded[start:end], label='Phase change from last sample')

# Adding title and labels
plt.title('Absolute')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.show()