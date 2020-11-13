import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# read data as 2D array of data type 'object'
data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

# read height and weight data into 2D array
X = data[:,0:2].astype(np.float)
X = X.T
X = X[:, X.min(axis=0) > 0]
X=X.T

#collecting heights
height_x=X[:,1:2].astype(np.float)

#collecting weights
weight_y=X[:,0:1].astype(np.float)

#collecting height^5
height_x_1 = [i ** 1 for i in height_x]
height_x_2 = [i ** 2 for i in height_x]
height_x_3 = [i ** 3 for i in height_x]
height_x_4 = [i ** 4 for i in height_x]
height_x_5 = [i ** 5 for i in height_x]
#converting array format into list
height_x_5 = [i.tolist() for i in height_x_5]
#converting list of lists into a single list
flat_list_height_x_5 = [item for sublist in height_x_5 for item in sublist]

#collecting height^10
height_x_6 = [i ** 6 for i in height_x]
height_x_7 = [i ** 7 for i in height_x]
height_x_8 = [i ** 8 for i in height_x]
height_x_9 = [i ** 9 for i in height_x]
height_x_10 = [i ** 10 for i in height_x]
#converting array format into list
height_x_10 = [i.tolist() for i in height_x_10]
#converting list of lists into a single list
flat_list_height_x_10 = [item for sublist in height_x_10 for item in sublist]

#calculating W1(D1),W5(D5),W10(D10) of the corresponding models
D1, b1,r_value, p_value, std_err = stats.linregress(height_x.T,weight_y.T)
D5, b5,r_value, p_value, std_err = stats.linregress(flat_list_height_x_5,weight_y.T)
D10, b10,r_value, p_value, std_err = stats.linregress(flat_list_height_x_10,weight_y.T)

#outlier height list
outliers_heights = [168., 172., 167.]
outliers_heights=np.array(outliers_heights).astype(np.float)

# resulting weights output for D=1,D=5, D=10
y1 = (D1 * outliers_heights) + b1
y5 =  (D5 * outliers_heights) + b5
y10 = (D10 * outliers_heights) + b10

#displaying outputs of weights for 3 models' 3 outliers
print(y1)
print(y5)
print(y10)

#plotting the figures
fig = plt.figure()
axs = fig.add_subplot(111)

#plotting data points
axs.plot(height_x,weight_y, 'ro', label='data')

#plotting outlier points for 3 models
axs.plot(outliers_heights,y1, 'bo', label='data')
axs.plot(outliers_heights,y5, 'yo', label='data')
axs.plot(outliers_heights,y10, 'go', label='data')

#plotting lines of the corresponding models'outliers
plt.plot(outliers_heights,y1)
plt.plot(outliers_heights,y5)
plt.plot(outliers_heights,y10)

#miscellenous plots
plt.legend(('Given_data_plots', 'Plotted_Line_D=1 for 3 outliers','Plotted_Line_D=5 " " "','Plotted_Line_D=10 " " "'))
plt.xlabel('height')
plt.ylabel('weight')
plt.title('weight,D=1,D=5,D=10', fontdict=None, loc='center', pad=None)
plt.savefig('2.1.pdf')
plt.show()
