import numpy as np
import random


def calc_prob(X, K, pMu, pSigma):
    N = X.shape[0]
    D = X.shape[1]
    Px = np.zeros((N, K))
    for i in range(K):
        Xshift = X-np.tile(pMu[i], (N, 1))
        lambda_flag = np.e**(-5)
        conv = pSigma[i]+lambda_flag*np.eye(D)
        inv_pSigma = np.linalg.inv(conv)
        tmp = np.sum(np.dot(Xshift, inv_pSigma)*Xshift, axis=1)
        coef = (2*np.pi)**(-D/2)*np.sqrt(np.linalg.det(inv_pSigma))
        Px[:, i] = coef*np.e**(-1/2*tmp)
    return Px


def gmm(X, K):       #用array来处理
    threshold = np.e**(-15)
    N = X.shape[0]
    D = X.shape[1]
    rndp = random.sample(np.arange(N).tolist(),K)
    centroids = X[rndp,:]
    pMu = centroids
    pPi = np.zeros((1, K))
    pSigma = np.zeros((K, D, D))
    dist = np.tile(np.sum(X*X, axis=1).reshape(N,1), (1, K))+np.tile(np.sum(pMu*pMu, axis=1), (N, 1))-2*np.dot(X, pMu.T)
    labels = np.argmin(dist,axis=1)
    for i in range(K):
        index = labels == i
        Xk = X[index,:]
        pPi[:,i] = (Xk.shape[0])/N
        pSigma[i] = np.cov(Xk.T)
    Loss = -float("inf")
    while True:
        Px = calc_prob(X, K, pMu, pSigma)
        pGamma = Px*np.tile(pPi, (N, 1))
        pGamma = pGamma/np.tile(np.sum(pGamma, axis=1).reshape(N,1), (1, K))
        Nk = np.sum(pGamma, axis=0)
        pMu = np.dot(np.dot(np.diag(1/Nk), pGamma.T), X)
        pPi = Nk/N
        for i in range(K):
            Xshift = X-np.tile(pMu[i], (N, 1))
            pSigma[i] = np.dot(Xshift.T, np.dot(np.diag(pGamma[:, i]), Xshift))/Nk[i]
        L = np.sum(np.log(np.dot(Px, pPi.T)), axis=0)
        if L-Loss < threshold:
            break
        Loss = L
    return Px,pMu,pSigma,pPi
        

if __name__ == "__main__":        
    Data_list = []
    with open("data.txt", 'r') as file:
        for line in file.readlines():  
            point = []  
            point.append(float(line.split()[0]))  
            point.append(float(line.split()[1]))  
            Data_list.append(point)  
    Data = np.array(Data_list)
    Px,pMu,pSigma,pPi = gmm(Data, 2)
    print(Px,pMu,pSigma,pPi)