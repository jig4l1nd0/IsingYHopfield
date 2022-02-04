from main import *


def main():
    random.seed(10)
    
    n=6 #range de i
    m=6 #range de j
    #p=10
    P=np.zeros([n,m])
    N=m*n
    T = 1
    k = 100    
        
        
    InitMat = [
               [[0, 1,	1,	1,	1,	0],
                [0,	1,	1,	1,	1,	0],
                [0,	1,	0,	0,	1,	0],
                [0,	1,	0,	0,	1,	0],
                [0,	1,	1,	1,	1,	0],
                [0,	1,	1,	1,	0,	0]]
              
              ,[[0, 1,	1,	1,	0,	0],
                [0,	0,	1,	1,	0,	0],
                [0,	0,	1,	1,	0,	0],
                [0,	0,	1,	1,	0,	0],
                [0,	1,	1,	1,	1,	0],
                [1,	1,	0,	0,	1,	1]]
              ]
        
    
    

    Conf = [lineal_matrix(mat) for mat in InitMat]      
    
    A = [a(config) for config in Conf] 
    f,W_pre = w(Conf,A)  
    W = f*W_pre    
    Th = Theta(W)
    plt.figure(1)
    plt.imshow(InitMat[0])
    
    lineal_data = lineal_matrix(InitMat[0]) 
    #print(lineal_data)
    Hu=H(lineal_data,n,m,W)
    
    energias,S,H_fin = Metropolis_Ising(lineal_data,Hu,T,n*m,Th,W,n)
                                           
    Uf=array(S).reshape(n,m);    
  
  
    plt.figure(2)
    plt.imshow(Uf)
    
    
    
    plt.figure(3)
    plt.plot( energias)
    #plt.imshow(W)
    plt.show()
    
    print("finisehd")
    return 0


if __name__ == "__main__":
    main()
