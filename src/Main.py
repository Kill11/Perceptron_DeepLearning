import numpy as np
from PIL import Image

#np.set_printoptions(threshold=np.nan)

matrix = [1]*50
for i in range(0,50):
    im = Image.open('../three/%s.png' %(i+1))
    im = im.convert("L")
    #im.show()
    matrix[i] = np.asarray(im)/255
    matrix[i] = matrix[i].reshape(-1,1)
    #matrix是一张图的列向量，数组

for i in range(0,49):
    if(i==0):
        x=matrix[i]
    x = np.hstack((x,matrix[i+1]))
    #all是30张图列向量的合并，矩阵

def sigmoid(x):
    # sigmoid function
    return 1/(1 + np.exp(-x))


y = np.zeros(50)+1
for q in range(30,50):
    y[q]=0
y = np.asmatrix(y)
x = np.asmatrix(x)
w = np.random.random((1024,1))
#w = np.zeros([1024,1])+0.3
w = np.asmatrix(w)
b = 0

#print(w)

for j in range(0,10001):
    z = np.dot(np.transpose(w),x)+b
    a = sigmoid(z/10)

    dz = a-y
    dw = 1/50*np.dot(x,np.transpose(dz))
    db = 1/50*np.sum(dz)
    w = w - 0.01*dw
    b = b - 0.01*db
# print(a)
# print(w)
# print(b)

im1 = Image.open('../three/test1.png')
im1 = im1.convert("L")
#im.show()
test = np.asarray(im1)/255
test = test.reshape(-1,1)
test = np.asmatrix(test)

z = np.dot(np.transpose(w),test)+b
a = sigmoid(z/10)
print(a)

im1 = Image.open('../three/test2.png')
im1 = im1.convert("L")
#im.show()
test = np.asarray(im1)/255
test = test.reshape(-1,1)
test = np.asmatrix(test)

z = np.dot(np.transpose(w),test)+b
a = sigmoid(z/10)
print(a)