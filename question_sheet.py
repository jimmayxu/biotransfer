
n = 10
def fib(n):
    n1 = 0
    n2 = 1
    for i in range(n):
        nth = n1 + n2
        n1 = n2
        n2 = nth
    return nth

fib(10)
n=3
i=3
import numpy as np
def sq(n, i):
    a = np.sqrt(n)
    a_ = np.floor(a*10**i)/10**i
    return a_

sq(3,5)


a = np.array([1,4,10,3,2,4,6,7])
import pandas as pd
i = 1


def incresaing_seq(a):
    a_sort = np.sort(a)
    for i, ai in enumerate(a_sort):
       if not any(pd.Series(ai).isin(a[i:])):
           print(ai)
np.arange(1,len(a)+1)
for i in range(len(a)):
    a[i:]

incresaing_seq(a)

