import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt

complex_numbers = np.fromfile("AAAAAAAAA.dat", dtype=np.complex64)

absolute = []
Dphi=[]
NRZI_decoded=[]

sample_rate = 1e6

# Design a high-pass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def highpass_filter(data, lowcut, highcut, sample_rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    y = filtfilt(b, a, data)
    return y

# Parameters
lowcut = 2000  # Desired cutoff frequency of the high-pass filter, Hz
highcut = 5000  # Desired cutoff frequency of the high-pass filter, Hz

# Apply the high-pass filter
dc_removed_signal = highpass_filter(complex_numbers, lowcut, highcut, sample_rate)

fft_result = np.fft.fft(dc_removed_signal)
magnitudes = np.abs(fft_result)
freqs = np.fft.fftfreq(len(dc_removed_signal),1/sample_rate)


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

Phase_change_cumulative = []
for i in range(len(Dphi)//100):
    if ((i+1)*100<=len(Dphi)):
        Phase_change_cumulative.append(sum(Dphi[i*100:(i+1)*100])*180/math.pi)


# Plot the spectrum
plt.figure(figsize=(10, 5))
plt.plot(freqs[:len(complex_numbers)], magnitudes[:len(complex_numbers)])  # Plotting only the positive half
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.show()

""" print(len(absolute))

# Define the range you want to plot, for example, the first 100 elements
start = 0
end = len(Dphi)

# Plotting
plt.figure(figsize=(10, 5))

# Plot real parts
plt.plot(Dphi2[start:end], label='Phase change from last sample')

# Adding title and labels
plt.title('Absolute')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.show() """