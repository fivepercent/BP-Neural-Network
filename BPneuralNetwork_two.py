class BPneuralNetwork_two():
    def __init__(self, num_output, num_input, lr, max_iter):
        self.num_output=num_output #number of outputs
        self.num_input=num_input #number of inputs
        self.lr=lr if lr!=None else 0.1 #learning rate
        self.max_iter=max_iter if max_iter!=None else 10 #max iteration times
        self.v1=np.random.rand(self.num_input,self.num_input)/100.0 #hidden weight 1
        self.v2=np.random.rand(self.num_input,self.num_input)/100.0 #hidden weight 2
        self.w=np.random.rand(self.num_output,self.num_input)/100.0 #output layer weight
        self.threshold_y=np.random.rand(1,self.num_output) #threshold for output layer
        self.threshold_h1=np.random.rand(1,self.num_input) #threshold for hidden layer 1
        self.threshold_h2=np.random.rand(1,self.num_input) #threshold for hidden layer 2
        
    def loadData(self, data):
        x=data[:,1:]/255
        y=data[:,0]
        
        return x,y
    
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
    
    def fit(self,x,y):
        m,n=x.shape #number of training samples
        
        y_nn=np.zeros((m,self.num_output)) #outputs from nn
        b1=np.zeros((1,n)) #output of hidden layer 1
        b2=np.zeros((1,n)) #output of hidden layer 2
        g=np.zeros((1,self.num_output)) #gradient descent of output layer weight
        e1=np.zeros((1,n)) #gradient descent of hidden weight 1
        e2=np.zeros((1,n)) #gradient descent of hidden weight 2
        
        num_iter=0 # iteration times
        
        y_train=np.zeros((m, self.num_output))
        for i in range(m):
            y_train[i][y[i]]=1.0
        
        while(num_iter<self.max_iter):
            num_iter+=1
            for i in range(m):
                x=x_train[i]
                # calculation for hidden layer 1
                alpha1=np.zeros(n)
                for j in range(n):
                    alpha1[j]=np.dot(x, self.v1[j])
                b1=self.sigmoid(alpha1-self.threshold_h1)

                # calculation for hidden layer 2
                alpha2=np.zeros(n)
                for j in range(n):
                    alpha2[j]=np.dot(b1, self.v2[j])
                b2=self.sigmoid(alpha2-self.threshold_h2)
                
                #calculation for output layer
                beta=np.zeros(self.num_output)
                for j in range(self.num_output):
                    beta[j]=np.dot(b2, self.w[j])
                y_nn[i]=self.sigmoid(beta-self.threshold_y)

                #calculation for gradient descent of output layer weight
                g=y_nn[i]*(1-y_nn[i])*(y_train[i]-y_nn[i]).reshape(1,self.num_output)
                
                #calculation for gradient descent of hidden weight 2
                e2=b2*(1-b2)*np.dot(self.w.T,g.T).T

                #calculation for gradient descent of hidden weight 1
                e1=b1*(1-b1)*np.dot(self.v2.T,e2.T).T
                
                #update weight and threshold of output layer
                self.w=self.w+self.lr*(np.dot(b2.T, g).T)
                self.threshold_y=self.threshold_y-self.lr*g

                #update weight and threshold of hidden layer 2
                self.v2=self.v2+self.lr*(np.dot(b1.reshape(n,1), e2).T)
                self.threshold_h2=self.threshold_h2-self.lr*e2
                
                #update weight and threshold of hidden layer 1
                self.v1=self.v1+self.lr*(np.dot(x.reshape(n,1), e1).T)
                self.threshold_h1=self.threshold_h1-self.lr*e1                
        return self
    
    def predict(self, sample):
        return np.where(sample==np.max(sample))[0][0]
    
    
    def score(self, test_data):
        x_test,y_test=self.loadData(test_data)
        
        m_test,n_test=x_test.shape
        
        y_nn_test=np.zeros((m_test,self.num_output))
        for i in range(m_test):
            x=x_test[i]
            
            # calculation for hidden layer 1
            alpha1=np.zeros(self.num_input)
            for j in range(self.num_input):
                alpha1[j]=np.dot(x, self.v1[j])
            b1=self.sigmoid(alpha1-self.threshold_h1)
            
            # calculation for hidden layer 2
            alpha2=np.zeros(self.num_input)
            for j in range(self.num_input):
                alpha2[j]=np.dot(b1, self.v2[j])
            b2=self.sigmoid(alpha2-self.threshold_h2)
            
            #calculation for output layer
            beta=np.zeros(self.num_output)
            for j in range(self.num_output):
                beta[j]=np.dot(b2, self.w[j])
            y_nn_test[i]=self.sigmoid(beta-self.threshold_y)
        
        right=0
        for i in range(m_test):
            if self.predict(y_nn_test[i])==y_test[i]:
                right+=1
    
        return right/m_test