#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:21:52 2020

@author: root
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:25:50 2020

@author: root
"""
# find common prefix


input=["abc","abcef","abce"]

def find_pre(input):
    minlen = min(map(len,input))
    out =''
    for i in range(minlen):
        p=input[0][i]
        for j in range(1, len(input)):
            if p!=input[j][i]:
                return out
            if j==len(input)-1 and p == input[j][i]:
                out +=p
    return out

find_pre(input)


#find next permutation



    
def findp2(x):
    ix=0
    while ix<len(x)-1:
        if x[ix]<x[ix+1]:
            ix +=1
        else:
            break
    
    if ix >len(x)-2 :
        return ix
    else:
        lenf = len(x[:ix+1])
        ix2= findp2(x[ix+1:])
        if ix2==0:
            return ix2+lenf-1 # since index remain in the x[:ix+1]
        else:
            return ix2+lenf   # since index go over in x[ix+1:]
def nextperm2(x):
    ix =findp2(x)
    if ix ==0:
        out =x[::-1]
    else:
        out = x[:]
        out[ix],out[ix-1]=out[ix-1], out[ix]
    return out    
    
    


x ='456321'#'456123'#'45612'#'123'#'321'#'312'#'321'#
x = list(x)
ix = findp2(x)
nextperm2(x)


############## rotating image

x=[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
]



        
        
def fill(x,ci,cj,cval, N):
    ni,nj =cj, (N-1)-ci
    
    if x[ni][nj]>0:
        nval=x[ni][nj]
        x[ni][nj]=-1*cval
        fill(x,ni,nj,nval, N )
    
        
def goover(x):
    N = len(x)
    for i in range(N):
        for j in range(N):            
            fill(x,i,j,x[i][j],N)
               
            
goover(x) 


#####most common words
x = "Bob hit a ball, the hit BALL flew far after it was hit."

x = x.split()

def rem_punc(x):
    punc = list("!?',;.")
    x = list(x)
    out=''
    for xi in x:
        if xi not in punc:
            out+=xi.lower()
    return out
x1 =list(map(rem_punc,x))

ux1 = set(x1)
outcount = [0]*len(ux1)
banned=["hit"]
for k,xi in enumerate(ux1):
    if xi is not banned:
        outcount[k]=x1.count(xi)
            
outcount.argmax()
