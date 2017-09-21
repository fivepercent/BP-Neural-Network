class BPneuralNetwork_one():
    def __init__(self, num_output, num_input, lr, max_iter):
        self.num_output=num_output #number of outputs
        self.num_input=num_input #number of inputs
        self.lr=lr if lr!=None else 0.1 #learning rate
        self.max_iter=max_iter if max_iter!=None else 10 #max iteration times
        self.v=np.random.rand(self.num_input,self.num_input)/100.0 #hidden weight
        self.w=np.random.rand(self.num_output,self.num_input)/100.0 #output layer weight
        self.threshold_y=np.random.rand(1,self.num_output) #threshold for output layer
        self.threshold_h=np.random.rand(1,self.num_input) #threshold for hidden layer
        
    def loadData(self, data):
        x=data[:,1:]/255
        y=data[:,0]
        
        return x,y
    
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
    
    def fit(self,x,y):
        m,n=x.shape #number of training samples
        
        y_nn=np.zeros((m,self.num_output)) #outputs from nn
        b=np.zeros((1,n)) #output of hidden layer
        g=np.zeros((1,self.num_output)) #gradient descent of output layer weight
        e=np.zeros((1,n)) #gradient descent of hidden weight
        
        num_iter=0 # iteration times
        
        y_train=np.zeros((m, self.num_output))
        for i in range(m):
            y_train[i][y[i]]=1.0
        
        while(num_iter<self.max_iter):
            num_iter+=1
            for i in range(m):
                x=x_train[i]
                alpha=np.zeros(n)
                # calculation for hidden layer
                for j in range(n):
                    alpha[j]=np.dot(x, self.v[j])
                b=self.sigmoid(alpha-self.threshold_h)
                #calculation for output layer
                beta=np.zeros(self.num_output)
                for j in range(self.num_output):
                    beta[j]=np.dot(b, self.w[j])
                y_nn[i]=self.sigmoid(beta-self.threshold_y)

                #calculation for gradient descent of output layer weight
                g=y_nn[i]*(1-y_nn[i])*(y_train[i]-y_nn[i]).reshape(1,self.num_output)
                #calculation for gradient descent of hidden weight
                e=b*(1-b)*np.dot(self.w.T,g.T).T

                #update weight and threshold of output layer
                self.w=self.w+self.lr*(np.dot(b.T, g).T)
                self.threshold_y=self.threshold_y-self.lr*g

                #update weight and threshold of hidden layer
                self.v=self.v+self.lr*(np.dot(x.reshape(n,1), e).T)
                self.threshold_h=self.threshold_h-self.lr*e
        return self
    
    def predict(self, sample):
        return np.where(sample==np.max(sample))[0][0]
    
    
    def score(self, test_data):
        x_test,y_test=self.loadData(test_data)
        
        m_test,n_test=x_test.shape
        
        y_nn_test=np.zeros((m_test,self.num_output))
        for i in range(m_test):
            x=x_test[i]
            alpha=np.zeros(self.num_input)
            # calculation for hidden layer
            for j in range(self.num_input):
                alpha[j]=np.dot(x, self.v[j])
            b=self.sigmoid(alpha-self.threshold_h)
            #calculation for output layer
            beta=np.zeros(self.num_output)
            for j in range(self.num_output):
                beta[j]=np.dot(b, self.w[j])
            y_nn_test[i]=self.sigmoid(beta-self.threshold_y)
        
        right=0
        for i in range(m_test):
            if self.predict(y_nn_test[i])==y_test[i]:
                right+=1
    
        return right/m_test