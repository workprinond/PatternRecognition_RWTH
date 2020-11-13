import numpy as np
import matplotlib.pyplot as plt
import math



#Loading Text
data = np.loadtxt('myspace.csv',dtype=np.object,comments='#',delimiter=',')

#Initializing Values
y = data[:, 1].astype(np.float)
h = np.array(y[y>0])
x = np.array([i for i in range(1,len(h)+1)])
N= sum(h)
k=1
alpha = 1
k_alpha_matrix = np.array([k,alpha]).transpose()

#function for calculating newton parameters
def newton(k,alpha):
 sum_log_alpha = np.sum([h* math.log(x) for h,x in zip(h,x)])
 di_by_alpha_tothepower_k = sum([h*((x)/alpha)**k for h,x in zip(h,x)])
 di_by_alpha_tothepower_k_prod_log_di_by_alpha = sum([h*((((x)/alpha)**k)*(math.log(x/alpha))) for h,x in zip(h,x)])
 di_by_alpha_tothepower_k_prod_log_di_by_alpha2 = sum([h*((((x)/alpha)**k)*(math.log(x/alpha)**2)) for h,x in zip(h,x)])

 dl_by_dk = (N/k) - (N * math.log(alpha)) + (sum_log_alpha) - (di_by_alpha_tothepower_k_prod_log_di_by_alpha)
 dl_by_dalpha = (k/alpha) *(di_by_alpha_tothepower_k - N)
 d2l_by_dk2 = (-N/(k**2)) - (di_by_alpha_tothepower_k_prod_log_di_by_alpha2)
 d2l_by_dalpha2 =  (k/(alpha**2)) * (N - (k+1)*(di_by_alpha_tothepower_k))
 d2l_by_dkdalpha = 1/alpha*(di_by_alpha_tothepower_k) + (k/alpha*(di_by_alpha_tothepower_k_prod_log_di_by_alpha)) - (N/alpha)

 A= np.array([[d2l_by_dk2,d2l_by_dkdalpha],[d2l_by_dkdalpha,d2l_by_dalpha2]])
 A = np.linalg.inv(A)
 B= np.array([-1*(dl_by_dk),-1*(dl_by_dalpha)]).transpose()
 return np.matmul(A,B)

#Iterating K and Alpha
for i in range(1,20):
 k_alpha_matrix = k_alpha_matrix + newton(k,alpha)
 k = k_alpha_matrix[0]
 alpha = k_alpha_matrix[1]

#Weibull fn
pdf_wbull_fn =sum(h) * (k / alpha) * ((x / alpha) ** (k - 1)) * np.exp(-1 *((x / alpha) ** k))

#Plots
plt.plot(x,h)
plt.plot(x,pdf_wbull_fn)
plt.legend(('Google_Data', 'Weibull_Distribution'))
plt.title('Exercise_1.3', fontdict=None, loc='center', pad=None)
plt.savefig('ex1_3.pdf')
plt.show()









