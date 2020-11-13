import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
import scipy.ndimage as img
from scipy import stats





#Binarize the image
def foreground2BinImg(f):
 d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - img.filters.gaussian_filter(f, sigma=1.00, mode='reflect')
 d = np.abs(d)
 m = d.max()
 d[d< 0.1*m] = 0
 d[d>=0.1*m] = 1
 return img.morphology.binary_closing(d)

imgName='lightning-3'
f = msc.imread(imgName+'.png', flatten=True).astype(np.float)
g = foreground2BinImg(f).astype(int)

s= np.array([1/(2**i) for i in range(1,len(g)-2)])
resolution = s * len(g)
resolution = np.array(resolution).astype(int)
resolution = np.array(resolution[resolution>0])



#Split a matrix into many submatrices
def split(array, nrows, ncols):
 r, h = array.shape
 return (array.reshape(h // nrows, nrows, -1, ncols)
         .swapaxes(1, 2)
         .reshape(-1, nrows, ncols))

#Count boxes with 1 value
def countboxes(l):
 count = 0
 list1= split(g,l,l)
 for i in list1:
  if (1 in i):
   count = count+1

 return count

# Miscllenous functions
countboxeslist = []
for i in resolution:
  countboxeslist.append(countboxes(i))

countboxeslist = np.log(countboxeslist)
s1 = s[9:]
s= s[0:9]

log_1_by_s = np.log(1/s)
log_1_by_stest=np.log(1/s1)
fig = plt.figure()
axs = fig.add_subplot(111)
axs.plot(log_1_by_s,countboxeslist, 'ro', label='data')

#Training the data
D, b,r_value, p_value, std_err = stats.linregress(log_1_by_s,countboxeslist)
print(countboxeslist)
#Value from Tree for test


y_new = (D * log_1_by_s) + b
y_new1 = (D * log_1_by_stest) + b
boxes = np.exp(y_new1)
print(boxes)
# print(y_new1)
#Plots
scale = np.exp(log_1_by_stest) ** (-1)
print(scale)
axs = fig.add_subplot(111)
axs.set_xlim(0, 20)
axs.set_ylim(0, 20)
plt.plot(log_1_by_s,y_new)
# plt.plot(log_1_by_stest,y_new1)
plt.legend(('Given_data_plots', 'Trained_Line'))
plt.xlabel('log 1/s')
plt.ylabel('log n')
plt.title('Exercise_1.5_lightning', fontdict=None, loc='center', pad=None)
plt.savefig('ex1_5_lightning.pdf')
plt.show()

