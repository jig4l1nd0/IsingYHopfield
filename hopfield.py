#################################################
#  Redes Neuronales 2022                        #
# 
#
#################################################
# Red hopfield con modelo de ising utilizando algoritmo de metropilis 

from math import exp
import numpy as np
from numpy import array, zeros, dot, zeros_like, ones_like, sign, ones
from numpy.random import choice
import matplotlib.pyplot as plt
import random

from tensorflow.keras.datasets import mnist


def lineal_matrix(G):
    out = list()
    for substring in G:
        for elem in substring:
              out.append(elem)
    return out 


def round_numb_mat(numb):
  trans_num = list()
  for row in numb:
      new_col = list()
      for element in row:
        new_col.append(round(element/256))
      trans_num.append(new_col)
  return trans_num 


def get_number_dataset():
    (trainX, trainy), (testX, testy) = mnist.load_data()
    Numbers_train= list()
    for number in range(10):  
        i=0
        boolean = True
        while i<len(trainy) and boolean:
            if trainy[i] == number:
                Numbers_train.append(round_numb_mat(trainX[i]))     
                bool = False
                break
        i+=1  

    Numbers_test= list()
    for number in range(10):  
        i=0
        boolean = True
        while i<len(testy) and boolean:
            if testy[i] == number:
                Numbers_test.append(round_numb_mat(testX[i]))     
                bool = False
                break
            i+=1  
    
    return Numbers_train, Numbers_test



def a(config):
      return np.array(config).mean()
  
def w(Conf,A): 
    Wheight_Mat = np.empty((len(Conf[0]),len(Conf[0]))) 
    for i in range(len(Conf[0])):
        for j in range(len(Conf[0])): 
            if i == j:
                Wheight_Mat[i][j] = 0;
            else:    
            #print([(Conf[mu][i]*Conf[mu][j]) for mu in len(Conf)])
                factor  = np.multiply(np.array(A),1-(-1)*np.array(A))
                Wheight_Mat[i][j] = np.sum(np.divide(np.array([ float(config[i]*config[j]) for config in Conf]),factor))

    plt.imshow(Wheight_Mat)      
    return  1/(len(Conf[0])), Wheight_Mat  


def Theta(W):
    theta = list()
    for i in range(len(W)):
        sum = 0    
        for j in range(len(W)):      
            sum += W[i,j]/2    
        theta.append(sum)
    
    return theta
    
# return [ print(W[i]) for i in len(W)]  
  #np.sum(W[i])*(1/2);

def H(s,n,m, W):  
    Th = Theta(W)
    #print(f"{Th=}")
    suma1=0
    suma2=0  
    for i in range(n):
        for j in range(m):
            print(f"{W[i,j]=}")
            print(f"{s[i]=}")
            print(f"{s[j]=}")
            suma1+=W[i,j]*s[i]*s[j]
            print(f"{Th[i]=}")
            print(f"{s[i]=}")
            suma2+=Th[i]*s[i]
    return 0.5*suma1+suma2 

def dH(S,i,n,m):
    S2=S.copy()
    if S2[i]==0:
        S2[i]=1
    else:
        S2[i]=0 
    return H(S2,n,m)-H(S,n,m)
    
def dH2(S,i,n,Th,W):
    suma = 0
    for j in range(n):
        suma+=W[i][j]*(S[i]-1/2)*S[j]
    suma+= -Th[i]*S[i]
    return suma

def Proba(DH,T):
    if DH==0:
        return 1
    else:
        return min(1,exp(-DH/T))  


def Metropolis_Ising(S,H1,T,iteraciones):
  energias = list()
  H_fin = H1.copy()
  for t in range(iteraciones):
    aleatorio=random.randint(0,len(S)-1)
    s=S[aleatorio]
    DH1=dH2(S,aleatorio)
    print(DH1)
    P=Proba(DH1,T)
    x=random.random()
    if x<P:
      if s==0:
        S[aleatorio]=1
      else:
          S[aleatorio]=0 
      H_fin+=DH1
    if H_fin < 0:
      break 
    
    energias.append(H_fin)
  
  #print(H_fin)
  return energias,S,H_fin    

def graph_hoppfield(data_matrix,n=28,m=28,T=1,k=28*28):    
  #f, axarr = plt.subplots(2,1) 
  #axarr[0].imshow(data_matrix)
  lineal_data = lineal_matrix(data_matrix) 
  #print(lineal_data)
  Hu=H(lineal_data,n,m)
  FinalU=Metropolis_Ising(lineal_data,Hu,T,k)
  Uf=array(FinalU[1]).reshape(n,m);    
  #GUF=axarr[1].imshow(Uf)
  #plt.show()  
  return Uf 

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
    lineal_data = lineal_matrix(InitMat) 
    print(lineal_data)
    Hu=H(lineal_data,n,m,W)
    energias,S,H_fin = Metropolis_Ising(lineal_data,Hu,T,k)
    Uf=array(S).reshape(n,m);    
  
    plt.imshow(Uf)
    
    
    
    
    print("finisehd")
    return 0





if __name__ == "__main__":
    main()
    