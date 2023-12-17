# Test code for IEEE course final project
# Fan Cheng, 2024
import minimatrix as mm

#测试Matrix类的各个操作：定义一个3×3的matrix mat，测试其各个函数都运行正确。

mat = mm.Matrix(data = [[1,2,3],[6,5,4],[7,8,9]])
mat1 = mm.Matrix(data = [[2,3,4,5],[5,6,7,2],[9,1,8,3]])

print(mat)

print(mat.shape())

print(mat.reshape((9,1)))
print(mat1.reshape((2,6)))

print(mat.dot(mat))
print(mat.dot(mat1))

print(mat.T())
print(mat1.T())

print(mat.sum())
print(mat.sum(axis = 0))
print(mat.sum(axis = 1))

print(mat.copy())

print(mat.Kronecker_product(mat1))

print(mat[1:,2:3])
print(mat1[1:3,:3])

mat1[1:3,:3] = mm.Matrix(data = [[1,2,3],[9,6,5]])
print(mat1)

print(mat.__pow__(3))

print(mat.__add__(mat))

print(mat.__sub__(mat))

print(mat.__len__())

print(mat.__Gauss_elimination__()[0])

print(mat.det())

mat2 = mm.Matrix(data = [[1,1,2],[-2,0,1],[0,-1,2]])
print(mat2.det())

print(mat2.inverse())

print(mat.rank())
print(mat2.rank())

#接下来是对 Matrix 类以外函数的测试
print(mm.I(6))

print(mm.narray((2,4)))
print(mm.narray((4,2),5))

A = mm.Matrix(data = [[1,2,9],[3,8,9],[0,7,2]])
print(mat)
print(mm.concatenate((mat,A),axis = 1))

def func(n):
   return  n * 2

F = mm.vectorize(func)
print(F(mat))

#通过arange()函数生成0到24（不包括24）的1×24维矩阵m24，测试reshape([3,8]), reshape([24,1]), reshape([4,6])，并输出。
m24 = mm.arange(0,24,1)
print(m24)
print(m24.reshape([3,8]))
print(m24.reshape([24,1]))
print(m24.reshape([4,6]))

#测试zeros(), 生成一个3×3维全0矩阵的并输出。测试zeros_like(m24)
print(mm.zeros((3,3)))
print(mm.zeros_like(m24))

#测试ones(), 生成一个3×3维全1矩阵的并输出。测试ones_like(m24)
print(mm.ones((3,3)))
print(mm.ones_like(m24))

#测试nrandom(), 生成一个3×3维的随机矩阵并输出。测试nrandom_like(m24)
print(mm.nrandom((3,3)))
print(mm.nrandom_like(m24))

#测试使用自定义的Matrix类解决最小二乘法问题：利用nrandom() 生成 m×n 的随机矩阵 X 以及 n×1 的随机向量 w，并生成m×1的零均值的随机向量e。计算得到 Y=Xw+e。利用自己实现的矩阵乘法、矩阵求逆等功能计算最小二乘法的估计  w, 并与 w 比较。测试中取 m=1000, n=100.
m,n = 1000,100
X = mm.nrandom((m,n)) 
w = mm.nrandom((n,1))
#接下来生成一个零均值的随机向量
e = [[mm.random.random()] for _ in range(m)]
ave = sum(i[0] for i in e)/m
for i in e:
    i[0] -= ave
e = mm.Matrix(data = e)

Y = X.dot(w).__add__(e)
w_ = X.T().dot(X).inverse().dot(X.T()).dot(Y)

print(w)
print(w_)
