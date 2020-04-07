import pathlib
from AffinityRegression import AffReg, Timer
from scipy.io import loadmat
import os
from random import sample
import numpy as np

timer_total = Timer()

#load the data
path = pathlib.Path(__file__).parent.absolute()
dataset = loadmat(os.path.join(path, 'data/pbmc5kdc.mat')) 

DD = dataset['D']
PP = dataset['Ppbmc5kdc']
YY = dataset['Ypbmc5kdc']

#split into training and testing
nsamples = PP.shape[0]    
indices = sample(range(nsamples),int(nsamples*0.8))
P_train = PP[indices,:]
Y_train = YY[:,indices]
P_test = np.delete(PP, indices, 0)
Y_test = np.delete(YY, indices, 1)

#create the object of AffReg
reg = AffReg()

#train the model
reg.fit( DD,P_train, Y_train, lamda = 0.001, rsL2 = 0, spectrumA = 1, spectrumB = 0.6 )

#predict test data
Y_pred = reg.predict(P_test)

#retrive trained Warning
W_pred = reg.get_W()

#calculate the correlation between Y_pred with ground truth, and plot
corr = reg.corPlot(Y_pred, Y_test)

time_hhmmss = timer_total.get_time_hhmmss()
print("Total time elapsed: %s\n" % time_hhmmss)
 