import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

y = data[:, 1].astype(np.float)

# Mean and Variance setup
Zeros= np.zeros(len(y))
sigma = np.mean(y)
mu = np.std(y)
#Setting up figure
fig1 = plt.figure()
axs1 = fig1.add_subplot(111)
axs1.plot(y,Zeros, 'bo', label='data')
axs1.set_xlim(y.min() - 10, y.max() + 10)
axs1.set_ylim(0.0, 0.10)
axs1.plot(norm.pdf(y, sigma,mu ))

x = np.linspace(150, 200, 1000)

#Pdf Function
y = norm.pdf(x, sigma, mu)

#Plots
plt.plot(x, y, color='orange')
plt.legend(('Data', 'Normal'))
plt.title('Exercise_1.2', fontdict=None, loc='center', pad=None)
plt.savefig('ex1_2.pdf')
plt.show()
