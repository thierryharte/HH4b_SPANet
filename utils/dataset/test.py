import numpy as np

a=np.zeros((2,5,5))
b=np.zeros((3,5,5))+1
c=np.zeros((3,5,5))+2

test=[]
test.append(a)
test.append(b)
test.append(c)

print(test)

conc=np.concatenate(test)

# print(conc)
print(conc.shape)