#Vito PAL

import os
import sys
import numpy as np
import csv
import time

from math import *
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, ExpSineSquared , DotProduct, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor


#check if raw data exists
infile = sys.argv[1]
if not infile:
    print "data file load failure."
    sys.exit(1)


######################
#Preprocessing data
######################


#remove repetition of rows in array
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#repeated experiment
for r in range(0,1):
    print "repetition",r,":"
    mat = np.genfromtxt(infile, delimiter=',', dtype='float')
    matAll = mat[:,1:3]
    matAll = np.insert(matAll, matAll.shape[1], mat[:, -5].transpose(), axis = 1)  # insert slack value to the last column
    matAll = np.insert(matAll, matAll.shape[1], mat[:, 35:67].transpose(), axis = 1)
    matAll = np.insert(matAll, matAll.shape[1], mat[:, -4:-1].transpose(), axis = 1)
    matAll = unique_rows(matAll)

    robust_scaler = RobustScaler()
    matAll = robust_scaler.fit_transform(matAll[:,:])
    #print matAll.shape


    #shuffle the data
    np.random.seed(int(time.time()*1000 % 4294967295))
    np.random.shuffle(matAll)

    #select training set and input set
    Sset = matAll[:250,:]
    E = matAll[250:,:]
    #print Sset.shape
    #print E.shape


    ##################
    #PAL Algorithm
    ##################


    #compute region union
    def union(l,u,newl,newu):
        low = []
        up = []
        for i in range(0,len(l)):
            if (newl[i] > u[i]) or (newu[i] < l[i]) or (l[i] + u[i] == 0):
                low.append(0)
                up.append(0)
            else:
                low.append(max(l[i],newl[i]))
                up.append(min(u[i],newu[i]))
        return low,up

    #gp regression
    #area kernel
    gp_kernel1 =  C(1.0, (1e-3, 1000)) * DotProduct(1.0,(1e-5, 1e5)) + WhiteKernel(0.1, (1e-3, 1000))
    #power kernel
    gp_kernel2 =  C(1.0, (1e-3, 1000)) * RationalQuadratic(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5))  + WhiteKernel(0, (1e-3, 1000))
    #delay kernel
    gp_kernel3 =  C(1.0, (1e-3, 1000)) * RationalQuadratic(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5))  + WhiteKernel(0, (1e-3, 1000))
    gpr1 = GaussianProcessRegressor(kernel=gp_kernel1,optimizer='fmin_l_bfgs_b',normalize_y=True)#,n_restarts_optimizer=2)
    gpr2 = GaussianProcessRegressor(kernel=gp_kernel2,optimizer='fmin_l_bfgs_b',normalize_y=True)
    gpr3 = GaussianProcessRegressor(kernel=gp_kernel3,optimizer='fmin_l_bfgs_b',normalize_y=True)


    ##### initialization #####
    P0 = [0] * len(E)   #pareto flag hash table
    N0 = [0] * len(E)   #non-pareto flag hash table
    U0 = [1] * len(E)   #unclassified flag hash table
    Sid = [0] * len(E)   #sampled flag hash table

    error = 0.001
    iteration = 0
    t = 100

    ##### repeat #####
    while (np.sum(U0) > 0)and(iteration <= t):
        iteration += 1
        Beta = 2 * log(3.0 * len(E) * ((pi * iteration) ** 2) / 0.0006)
        #print "PAL iteration",iteration,"with beta",Beta,":"

        ##### modeling starts #####
        stime = time.time()
        #predict area
        gp1 = gpr1.fit(Sset[:,:-3],Sset[:,-3])
        areaP, areaU = gpr1.predict(E[:,:-3],return_std = True)
        #redict power
        gp2 = gpr2.fit(Sset[:,:-3],Sset[:,-2])
        powerP, powerU = gpr2.predict(E[:,:-3],return_std = True)
        #predict delay
        gp3 = gpr3.fit(Sset[:,:-3],Sset[:,-1])
        delayP, delayU = gpr3.predict(E[:,:-3],return_std = True)

        for i in range(0,len(E)):
            if (Sid[i] == 1):
                areaP[i], areaU[i] = E[i,-3], 0
                powerP[i], powerU[i] = E[i,-2], 0
                delayP[i], delayU[i] = E[i,-1], 0

        #Rt uncertainty region
        aLow , aUp = areaP - 0.001* sqrt(Beta) * areaU, areaP +  0.001* sqrt(Beta) * areaU
        pLow , pUp = powerP -  0.001* sqrt(Beta) * powerU, powerP +  0.001* sqrt(Beta) * powerU
        dLow , dUp = delayP -  0.001* sqrt(Beta) * delayU, delayP +  0.001* sqrt(Beta) * delayU

        if iteration == 1:
            RaLow, RaUp = aLow, aUp
            RpLow, RpUp = pLow, pUp
            RdLow, RdUp = dLow, dUp
        else:
            RaLow, RaUp = union(RaLow, RaUp, aLow, aUp)
            RpLow, RpUp = union(RpLow, RpUp, pLow, pUp)
            RdLow, RdUp = union(RdLow, RdUp, dLow, dUp)

        #print "modeling time:", time.time()-stime
        ##### modeling ends #####


        ##### classification starts #####
        stime = time.time()
        Pt = P0
        Nt = N0
        Ut = U0
        for x in range(0,len(E)):
            if (Ut[x] == 1):
                pareto = True
                nonpareto = False
                for y in range(0,len(E)):
                    if (x != y) and (Ut[y] == 1):
                        if (RaLow[y] * (1 + error) <= RaUp[x] * (1 - error)) and\
                           (RpLow[y] * (1 + error) <= RpUp[x] * (1 - error)) and\
                           (RdLow[y] * (1 + error) <= RdUp[x] * (1 - error)):
                            pareto = False
                        if (RaUp[y] * (1 - error) <= RaLow[x] * (1 + error)) and\
                           (RpUp[y] * (1 - error) <= RpLow[x] * (1 + error)) and\
                           (RdUp[y] * (1 - error) <= RdLow[x] * (1 + error)):
                            nonpareto = True
                            break
                if pareto:
                    Pt[x] = 1
                    Ut[x] = 0
                elif nonpareto:
                    Nt[x] = 1
                    Ut[x] = 0
        #print "classification time:", time.time()-stime
        ##### classification ends #####


        ##### sampling starts #####
        stime = time.time()
        maxwt = 0
        maxid = -1
        for x in range(0,len(E)):
            if ((Ut[x] == 1) or (Pt[x] == 1)) and not(Sid[x] == 1):
                wt = sqrt((RaUp[x]-RaLow[x])**2 +\
                          (RpUp[x]-RpLow[x])**2 +\
                          (RdUp[x]-RdLow[x])**2 )
                if maxid == -1:
                    maxwt = wt
                    maxid = x
                elif wt > maxwt:
                    maxwt = wt
                    maxid = x
        Sset = np.insert(Sset, Sset.shape[0], E[maxid], axis = 0)
        Sid[maxid] = 1
        P0 = Pt
        N0 = Nt
        U0 = Ut
        #print "sampling time:", time.time()-stime
        #print "Predicted Pareto set size: ", np.sum(Pt)
        #print "Predicted nonPareto set size: ", np.sum(Nt),"\n"
        ##### sampling ends #####


    ##################
    #Output results
    ##################

    print len(Sset)
    #check if output folder exists
    path = 'results/'
    if not os.path.exists(path):
        os.makedirs(path)


    #output predicted pareto set
    outputMat = robust_scaler.inverse_transform(E)
    out = 'result' + str(r) + '.csv'
    outPath = os.path.join(path, out)
    outputFile = file(outPath, 'wb')
    outFile_writer = csv.writer(outputFile)
    for i in range(0,len(Pt)):
        if (Pt[i] == 1):
            outFile_writer.writerow(outputMat[i][-3:])

    #close output file
    outputFile.close()
