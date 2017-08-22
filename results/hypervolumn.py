#vito PF
import numpy as np
import sys
import csv

#minimize first element, minimize second element
def pf(a):
    pf = []
    i = 0
    while i < len(a):
    #find a[i] in the pareto front
        j = 0
        while j < len(a):
            if i != j:  # to compare
                vj1 = a[j][0]
                vj2 = a[j][1]
                vi1 = a[i][0]
                vi2 = a[i][1]
                #if a[j] dominates a[i]
                if (vj1 <= vi1 and vj2 <= vi2) and (vj1 < vi1 or vj2 < vi2):
                    i += 1
                    break
                else: # increase j to compare next
                    j +=1
                    if j == len(a): # if no next, a[i] is in the PF
                        pf.append(a[i])
                        i += 1
                        break
            else: # increase j to compare next
                j += 1
                if i == len(a)-1 and j == len(a): # if no next, a[i] is in the PF
                    pf.append(a[i])
                    i += 1
    return pf

def Areaad(array):
    a = 0
    for i in range(0,len(array)-1):
        a += (array[i+1,1] - array[i,1]) * (maxarea - array[i,0])
    a += (maxdelay-array[-1,1]) * (maxarea - array[-1,0])
    return a

def Areapd(array):
    a = 0
    for i in range(0,len(array)-1):
        a += (array[i+1,1] - array[i,1]) * (maxpower - array[i,0])
    a += (maxdelay-array[-1,1]) * (maxpower - array[-1,0])
    return a
#output("ADPF_PAL50.csv",a)
#output("PDPF_PAL50.csv",p)

#max features in physical design space
maxdelay = 0.4112
maxarea = 2257.079015
maxpower = 8100.0


sumPAL,sumAlpha = 0,0
repetition = 1
maxadarea = 0
maxpdarea = 0
adareaPAL = [0] * repetition
pdareaPAL = [0] * repetition
adareaAlpha = [0] * repetition
pdareaAlpha = [0] * repetition
for i in range(0,repetition):
    #PAL
    matPAL = np.genfromtxt(('result'+str(i)+'.csv'), delimiter=',', dtype='float')
    sumPAL += len(matPAL)
    pdPAL = matPAL[:,1:]
    adPAL = np.delete(matPAL, 1, 1)
    pdPAL = np.array(pf(pdPAL))
    adPAL = np.array(pf(adPAL))
    adPAL = adPAL[adPAL[:,1].argsort()]
    pdPAL = pdPAL[pdPAL[:,1].argsort()]
    adareaPAL[i] = Areaad(adPAL)
    pdareaPAL[i] = Areapd(pdPAL)

    #Alpha
    #matadAlpha = np.genfromtxt(('prefixAreaDelay'+str(i)+'.csv'), delimiter=',', dtype='float')
    #matpdAlpha = np.genfromtxt(('prefixPowerDelay'+str(i)+'.csv'), delimiter=',', dtype='float')
    #matAlpha= np.concatenate((matadAlpha,matpdAlpha),axis = 0)
    #matAlpha = matAlpha[:,-4:-1]
    #sumAlpha += len(matAlpha)
    #pdAlpha = matAlpha[:,1:]
    #adAlpha = np.delete(matAlpha, 1, 1)
    #adAlpha = np.array(pf(adAlpha))
    #pdAlpha = np.array(pf(pdAlpha))
    #adAlpha = adAlpha[adAlpha[:,1].argsort()]
    #pdAlpha = pdAlpha[pdAlpha[:,1].argsort()]
    #adareaAlpha[i] = Areaad(adAlpha)
    #pdareaAlpha[i] = Areapd(pdAlpha)



print sumPAL * 1.0 / repetition + 250, sumAlpha * 1.0/ repetition + 693
print np.mean(adareaPAL), np.std(adareaPAL), np.max(adareaPAL), np.argmax(adareaPAL)
print np.mean(adareaAlpha), np.std(adareaAlpha), np.max(adareaAlpha), np.argmax(adareaAlpha),"\n"
print np.mean(pdareaPAL), np.std(pdareaPAL), np.max(pdareaPAL), np.argmax(pdareaPAL)
print np.mean(pdareaAlpha), np.std(pdareaAlpha), np.max(pdareaAlpha), np.argmax(pdareaAlpha),"\n"
