import cupy as cp
import cupyx
from mnist import load_mnist
import random


# def binary_sampler(threshold):
#     return cp.random.binomial(1,threshold)


# MNIST load
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False,flatten=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)
print(t_train)

with cp.cuda.Device(0): 
    #cp.random.seed(0)# シード値を設定
    rg = cp.random.default_rng()
    D = 400
    N1 = 28*28 # = 784
    N2 = 500
    N3 = 10
    M = 50

    # Hyper Parameters for learning                                                                                   
    rb = 1.0
    rq = 1.0

    rx = 1.0
    rs = 0.001
    a0 = 0.5



    # Visible Neurons
    x1 = x_train[0:D]
    x1 = cp.asarray(x1)
    x3 = t_train[0:D]
    x3 = cp.asarray(x3)

    # Hidden Neurons
    x2 = cp.random.Generator.binomial(rg, 1, 1/2, (D, N2))

    #x1 = cp.random.Generator.binomial(rg, 1, 1/2,(D, N1))
    #x2 = cp.zeros((D, N2))
    #x3 = cp.zeros((D, N3))


    # Synapse Connectivity
    m =  cp.arange(1, M+1)
    a = ( 2*(m - 1)/(M - 1) - 1) * a0

    s12 = cp.random.Generator.binomial(rg, 1, 1/2, (N1, N2, M))
    s23 = cp.random.Generator.binomial(rg, 1, 1/2, (N2, N3, M))

    #s12 = cp.zeros((N1, N2, M))
    #s23 = cp.zeros((N2, N3, M))

    # Weights
    w12 = cp.squeeze(cp.dot(s12, a)) #N1*N2
    w23 = cp.squeeze(cp.dot(s23, a)) #N2*N3


    # Gibbs Sampling
    ITR = 3000
    ITR = 10
    for itr in range(ITR):
        #Neurons Update
        for j in range(N2):
            # Weights
            w12 = cp.squeeze(cp.dot(s12, a)) #N1*N2
            w23 = cp.squeeze(cp.dot(s23, a)) #N2*N3
            vj = cp.dot(x1,w12[:,j]) #D*1
            vk = cp.dot(x2,w23)      #D*N3
            tmpk = x3 - cupyx.scipy.special.expit(vk)
            wjk =cp.tile(w23[j,:],(D,1)) #D*N3
            bj = cp.sum(wjk*tmpk,axis=1) #D*1
            x2[:,j] = cp.random.Generator.binomial(rg, 1, cupyx.scipy.special.expit (vj+bj))
            pass

        #Synapses Update
        for j in range(N2):
            w12 = cp.squeeze(cp.dot(s12, a)) #N1*N2
            vj = cp.dot(x1,w12[:,j]) #D*None
            tmpd = x2[:,j] - cupyx.scipy.special.expit(vj) # D*None
            for i in range(N1):
                qij = cp.sum(x1[:,i]*tmpd)
                s12[i,j,:] = cp.random.Generator.binomial(rg, 1, cupyx.scipy.special.expit (a*qij))
                pass
        for j in range(N3):
            w23 = cp.squeeze(cp.dot(s23, a)) #N2*N3
            vj = cp.dot(x2,w23[:,j]) #D*None
            tmpd = x3[:,j] - cupyx.scipy.special.expit(vj) # D*None
            for i in range(N2):
                qij = cp.sum(x2[:,i]*tmpd)
                s23[i,j,:] = cp.random.Generator.binomial(rg, 1, cupyx.scipy.special.expit (a*qij))
                pass

        #feed forward
        correct =cp.zeros((D))
        pred    =cp.zeros((D))
        cp.argmax(x3,axis=1,out=correct)
        pw12 = cp.squeeze(cp.dot(s12, a)) #N1*N2
        pw23 = cp.squeeze(cp.dot(s23, a)) #N2*N3    
        px2 = cp.dot(x1,cp.squeeze(pw12))
        px2 = cp.random.Generator.binomial(rg, 1, cupyx.scipy.special.expit (px2))
        px3 = cp.dot(px2,cp.squeeze(pw23))
        #print(px3)
        cp.argmax(px3,axis=1,out=pred)
        #x3 = cp.random.Generator.binomial(rg, 1, cupyx.scipy.special.expit (x3))
        print("accuracy:"+str( cp.count_nonzero(correct == pred)/D))
    print(correct)
    print(pred)
    pass
