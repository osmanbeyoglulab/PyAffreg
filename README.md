# PyAffreg  <span style="font-size:smaller;">(Affinity regression  Python/Cython/C implementation)</span>

### About AffReg Class
**Class name**: AffReg. 

**Class descrition**: Affinity regression(AffReg) explains the interaction between two objects.

    D * W * P.T ~ Y
D is a given object feature, P is the other object feature, Y is the interaction between D and P.

After taining the model, We want to predict the interaction between D and new features of P.

**Class functions**: 

	fit(D,  P_train, Y_train, lamda = 0.001, rsL2 = 0, spectrumA = 1 spectrumB = 0.7, norm=True )
	# train the model 
	# parameters:		
		D: left matrix
		P_train: right matrix, train sample of P
		Y_train: interation, train sample of Y
		lamda: L1 regulization
		rsL2:  L2 regulization
		sectrumA: percent spectrum to keep (0,1] for left matrix
		sectrumB: percent spectrum to keep (0,1] for right matrix
	
	predict(P_test)
	# predidct the interaction on test samples of P
        # parameters:
		P_test: test sample of P. 
	
	get_W()
        # retrieve trained parameters W
        
	corPlot (Y_pred, Y_test)
	# calculate the correlation between Y_pred with ground truth, and plot the their correlation
	# parameters:
		Y_pred: predicted interaction from predict function, 
	    	Y_test: the ground truth value of Y, usually it is the test dataset from data splitting
					
### How to run

#### 1. Download PyAffreg
#### 2. Build Bython extension module

Affreg calls a few cython customerized extensions which need to be built on site

** - build cythLeastR extention modue **
    
1)  Goto folder "cythKrnPlus built", then execute command:
 
        python setup.py build_ext --inplace

It will generate the file cythLeastR.cpython-37m-x86_64-linux-gnu.so (Mac or Linux), or cythLeastR.cp37-win_amd64.pyd (Windows). 
    
2) Copy the file into pyAffreg folder

** - build cythKrnPlus extention modue **
    
1) Goto folder "cythKrnPlus built", then execute command:

    python setup.py build_ext --inplace
    
It will generate the file cythKrnPlus.cpython-37m-x86_64-linux-gnu.so (Mac or Linux), or cythKrnPlus.cp37-win_amd64.pyd (Windows). 
    
2) Copy the file into pyAffreg folder

#### 3. Use AffReg class

1) Import the class
        
        from AffinityRegression import AffReg

2) Define the object of the class
        
        reg = AffReg()

3) Train the model
        
        reg.fit(D,  P_train, Y_train, lamda = 0.001, rsL2 = 0, spectrumA = 1 spectrumB = 0.7, norm=True )
        # lamda, reL2, spectrumA, spectrumB can be changed, aboves are the default falue
4) Predict Y
        
        pred_Y = reg.predict(P_test)
   After training the model, you can also retrieve the trained parameter W with
        
        pred_W = reg.get_W()
5) Check the performance and plot the correlation
        
        corr = reg.corPlot(Y_pred, Y_test)


