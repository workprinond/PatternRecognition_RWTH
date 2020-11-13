import matplotlib.pyplot as plt
from numpy import array, linalg, random

#Unit Circle Plot
def plotUnitCircle(p):
 for i in range(100000):
  x = array([random.rand()*2-1,random.rand()*2-1])
  if linalg.norm(x,p) < 1:
   plt.plot(x[0],x[1],'go')
 plt.axis([-1.5, 1.5, -1.5, 1.5])
 plt.title('Exercise_1.4(p value = 0.5)', fontdict=None, loc='center', pad=None)
 plt.savefig('ex1_4.pdf')
 plt.show()

#Calling Function for p = 1/2
plotUnitCircle(1/2)

