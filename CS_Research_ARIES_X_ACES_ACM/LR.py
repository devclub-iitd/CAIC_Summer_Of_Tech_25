import numpy as np
import pandas as pd

def grad(x:np.array,y:np.array,w:np.array,b:float,lambda_:float,n:int):
    
    #calculate grad
    pred = w * x + b                  
    e = pred - y                  
    dc_dw = np.sum(e * x) / n  + (lambda_ / n) * w
    dc_db = np.sum(e) / n       

    return dc_dw,dc_db

def linearRegression(X: np.array, Y: np.array, lr: float, lambda_: float):
    """
    Parameters:
    - X: Input feature matrix (NumPy array)
    - Y: Target vector (NumPy array)
    - lr: Learning rate (float)
    - lambda_: L2 regularization coefficient (float)

    Returns:
    - weights: Learned model parameters
    """
    
    # assumin Y,X - nx1
    n=X.shape[0]
    w = 0.01*np.random.rand()  
    b = 0.01*np.random.rand()
    cost_hist=[]
    w_hist=[]
    b_hist=[]
    
    X= (X - X.min()) / (X.max() - X.min())
    Y= (Y - Y.min()) / (Y.max() - Y.min())
    

    for i in range(1000):
        #calculate cost
        #calculate cost
        f_wb = w * X + b   
        reg_term = (lambda_ / (2 * n)) * np.sum(w ** 2)     #only L2 reg.
        cost = np.sum((f_wb - Y) ** 2)+reg_term
        #update
        dc_dw,dc_db= grad(X,Y,w,b,lambda_,n)
        w= w - lr*dc_dw
        b= b- lr*dc_db

        cost_hist.append(cost)
        w_hist.append(w)
        b_hist.append(b)

        if i%10 ==0:
            print(f"Iteration {i:4}: Cost {float(cost_hist[-1]):8.2f} ")

    return w,b

    pass
