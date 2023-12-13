# Framework for IEEE course final project
# Fan Cheng, 2022

import random

class Matrix:
	r"""
	自定义的二维矩阵类

	Args:
		data: 一个二维的嵌套列表，表示矩阵的数据。即 data[i][j] 表示矩阵第 i+1 行第 j+1 列处的元素。
			  当参数 data 不为 None 时，应根据参数 data 确定矩阵的形状。默认值: None
		dim: 一个元组 (n, m) 表示矩阵是 n 行 m 列, 当参数 data 为 None 时，根据该参数确定矩阵的形状；
			 当参数 data 不为 None 时，忽略该参数。如果 data 和 dim 同时为 None, 应抛出异常。默认值: None
		init_value: 当提供的 data 参数为 None 时，使用该 init_value 初始化一个 n 行 m 列的矩阵，
					即矩阵各元素均为 init_value. 当参数 data 不为 None 时，忽略该参数。 默认值: 0
	
	Attributes:
		dim: 一个元组 (n, m) 表示矩阵的形状
		data: 一个二维的嵌套列表，表示矩阵的数据

	Examples:
		>>> mat1 = Matrix(dim=(2, 3), init_value=0)
		>>> print(mat1)
		>>> [[0 0 0]
			 [0 0 0]]
		>>> mat2 = Matrix(data=[[0, 1], [1, 2], [2, 3]])
		>>> print(mat2)
		>>> [[0 1]
			 [1 2]
			 [2 3]]
	"""
	def __init__(self, data=None, dim=None, init_value=0):
		if not (data or dim):
			print('Bad assignment!')
			raise ValueError 
		self.data = data
		self.dim = (len(data), len(data[0]))
		

	def shape(self):
		r"""
		返回矩阵的形状 dim
		"""
		return self.dim

	def reshape(self, newdim):
		r"""
		将矩阵从(m,n)维拉伸为newdim=(m1,n1)
		该函数不改变 self
		
		Args:
			newdim: 一个元组 (m1, n1) 表示拉伸后的矩阵形状。如果 m1 * n1 不等于 self.dim[0] * self.dim[1],
					应抛出异常
		
		Returns:
			Matrix: 一个 Matrix 类型的返回结果, 表示 reshape 得到的结果
		"""
		if newdim[0] * newdim[1] != self.dim[0] * self.dim[1]: 
			print('Inapropriate value was assigned to newdim!')
			raise ValueError
		row_altogether = (self.data[i][j] for i in range(self.dim[0]) for j in range(self.dim[1]))
		newdata = [[next(row_altogether) for _ in range(newdim[1])] for _ in range(newdim[0])]
		newmatrix = Matrix(data=newdata, dim=newdim)
		return newmatrix

	def dot(self, other):
		r"""
		矩阵乘法：矩阵乘以矩阵
		按照公式 A[i, j] = \sum_k B[i, k] * C[k, j] 计算 A = B.dot(C)

		Args:
			other: 参与运算的另一个 Matrix 实例
		
		Returns:
			Matrix: 计算结果
		
		Examples:
			>>> A = Matrix(data=[[1, 2], [3, 4]])
			>>> A.dot(A)
			>>> [[ 7 10]
				 [15 22]]
		"""
		dot_data = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.dim[1])) for j in range(other.dim[1])] for i in range(self.dim[0])]
		dot_mat = Matrix(data=dot_data, dim=(self.dim[0], other.dim[1]))
		return dot_mat

	def T(self):
		r"""
		矩阵的转置

		Returns:
			Matrix: 矩阵的转置

		Examples:
			>>> A = Matrix(data=[[1, 2], [3, 4]])
			>>> A.T()
			>>> [[1 3]
				 [2 4]]
			>>> B = Matrix(data=[[1, 2, 3], [4, 5, 6]])
			>>> B.T()
			>>> [[1 4]
				 [2 5]
				 [3 6]]
		"""
		transpose_data = [[self.data[i][j] for i in range(self.dim[0])] for j in range(self.dim[1])]
		transpose_mat = Matrix(data=transpose_data, dim=(self.dim[1], self.dim[0]))
		return transpose_mat

	def sum(self, axis=None): 
		r"""
		根据指定的坐标轴对矩阵元素进行求和

		Args:
			axis: 一个整数，或者 None. 默认值: None
				  axis = 0 表示对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
				  axis = 1 表示对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
				  axis = None 表示对矩阵全部元素进行求和，得到形状为 (1, 1) 的矩阵
		
		Returns:
			Matrix: 一个 Matrix 类的实例，表示求和结果

		Examples:
			>>> A = Matrix(data=[[1, 2, 3], [4, 5, 6]])
			>>> A.sum()
			>>> [[21]]
			>>> A.sum(axis=0)
			>>> [[5 7 9]]
			>>> A.sum(axis=1)
			>>> [[6]
				 [15]]
		"""
		match axis:
			case None:
				return Matrix(
					data=[[sum(self.data[i][j] for i in range(self.dim[0]) for j in range(self.dim[1]))]]\
					,dim=(1, 1)
				  )
			case 0:
				return Matrix(
					data=[[sum(self.data[i][j] for i in range(self.dim[0])) for j in range(self.dim[1])]]\
					,dim=(1, self.dim[1])
				)
			case 1:
				return Matrix(
					data=[[sum(self.data[i][j] for j in range(self.dim[1]))] for i in range(self.dim[0])]\
					,dim=(self.dim[0], 1)
				)
			case _:
				print('Wrong input, which should be in (None, 1, 0)')
				raise ValueError

	def copy(self):
		r"""
		返回matrix的一个备份

		Returns:
			Matrix: 一个self的备份
		"""
		return Matrix(data=self.data, dim=self.dim)

	def Kronecker_product(self, other):
		r"""
		计算两个矩阵的Kronecker积,具体定义可以搜索,https://baike.baidu.com/item/克罗内克积/6282573

		Args:
			other: 参与运算的另一个 Matrix

		Returns:
			Matrix: Kronecke product 的计算结果
		"""
		Kronecker_data = [[self.data[i][j] * other.data[m][n] for n in range(other.dim[1]) for j in range(self.dim[1])]\
					for m in range(other.dim[0]) for i in range(self.data[0])]
		Kronecker_mat = Matrix(data=Kronecker_data, dim=(self.dim[0] * other.dim[0], self.dim[1] * other.dim[1]))
		return Kronecker_mat
	
	def __getitem__(self, key):
		r"""
		实现 Matrix 的索引功能，即 Matrix 实例可以通过 [] 获取矩阵中的元素（或子矩阵）

		x[key] 具备以下基本特性：
		1. 单值索引
			x[a, b] 返回 Matrix 实例 x 的第 a 行, 第 b 列处的元素 (从 0 开始编号)
		2. 矩阵切片
			x[a:b, c:d] 返回 Matrix 实例 x 的一个由 第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵
			特别地, 需要支持省略切片左(右)端点参数的写法, 如 x 是一个 n 行 m 列矩阵, 那么
			x[:b, c:] 的语义等价于 x[0:b, c:m]
			x[:, :] 的语义等价于 x[0:n, 0:m]

		Args:
			key: 一个元组，表示索引

		Returns:
			索引结果，单个元素或者矩阵切片

		Examples:
			>>> x = Matrix(data=[
						[0, 1, 2, 3],
						[4, 5, 6, 7],
						[8, 9, 0, 1]
					])
			>>> x[1, 2]
			>>> 6
			>>> x[0:2, 1:4]
			>>> [[1 2 3]
				 [5 6 7]]
			>>> x[:, :2]
			>>> [[0 1]
				 [4 5]
				 [8 9]]
		"""
		a, b = key
		if type(a) is int:
			return self.data[a][b]
		else:
			slice_data = [row[b] for row in self.data[a]]
		return Matrix(data=slice_data)

		

	def __setitem__(self, key, value):
		r"""
		实现 Matrix 的赋值功能, 通过 x[key] = value 进行赋值的功能

		类似于 __getitem__ , 需要具备以下基本特性:
		1. 单元素赋值
			x[a, b] = k 的含义为，将 Matrix 实例 x 的 第 a 行, 第 b 处的元素赋值为 k (从 0 开始编号)
		2. 对矩阵切片赋值
			x[a:b, c:d] = value 其中 value 是一个 (b-a)行(d-c)列的 Matrix 实例
			含义为, 将由 Matrix 实例 x 的第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵 赋值为 value 矩阵
			即 子矩阵的 (i, j) 位置赋值为 value[i, j]
			同样地, 这里也需要支持如 x[:b, c:] = value, x[:, :] = value 等省略写法
		
		Args:
			key: 一个元组，表示索引
			value: 赋值运算的右值，即要赋的值

		Examples:
			>>> x = Matrix(data=[
						[0, 1, 2, 3],
						[4, 5, 6, 7],
						[8, 9, 0, 1]
					])
			>>> x[1, 2] = 0
			>>> x
			>>> [[0 1 2 3]
				 [4 5 0 7]
				 [8 9 0 1]]
			>>> x[1:, 2:] = Matrix(data=[[1, 2], [3, 4]])
			>>> x
			>>> [[0 1 2 3]
				 [4 5 1 2]
				 [8 9 3 4]]
		"""
		a, b = key
		if type(a) is int:
			if type(value) is not int:
				raise ValueError
			self.data[a][b] = value
		else:
			if type(value) is not Matrix:
				raise ValueError
			for i in range(len(value.data)):
				self.data[a][i][b] = value.data[i]
		return

	def __pow__(self, n):
		r"""
		矩阵的n次幂,n为自然数
		该函数应当不改变 self 的内容

		Args:
			n: int, 自然数

		Returns:
			Matrix: 运算结果
		"""
		res = self
		for _ in range(n):
			res = res.dot(self)
		return res

	def __add__(self, other):
		r"""
		两个矩阵相加
		该函数应当不改变 self 和 other 的内容

		Args:
			other: 一个 Matrix 实例
		
		Returns:
			Matrix: 运算结果
		"""
		add_data = [[self.data[i][j] + other.data[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])]
		return Matrix(data=add_data, dim=self.dim)

	def __sub__(self, other):
		r"""
		两个矩阵相减
		该函数应当不改变 self 和 other 的内容

		Args:
			other: 一个 Matrix 实例
		
		Returns:
			Matrix: 运算结果
		"""
		sub_data = [[self.data[i][j] - other.data[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])]
		return Matrix(data=sub_data, dim=self.dim)

	def __mul__(self, other):
		r"""
		两个矩阵 对应位置 元素  相乘
		注意 不是矩阵乘法dot
		该函数应当不改变 self 和 other 的内容

		Args:
			other: 一个 Matrix 实例
		
		Returns:
			Matrix: 运算结果

		Examples:
			>>> Matrix(data=[[1, 2]]) * Matrix(data=[[3, 4]])
			>>> [[3 8]]
		"""
		mul_data = [[self.data[i][j] * other.data[i][j] for j in range(self.dim[1])] for i in range(self.dim[0])]
		return Matrix(data=mul_data, dim=self.dim)


	def __len__(self):
		r"""
		返回矩阵元素的数目

		Returns:
			int: 元素数目，即 行数 * 列数
		"""
		return self.dim[0] * self.dim[1]

	def __str__(self):
		r"""
		按照
		[[  0   1   4   9  16  25  36  49]
 		 [ 64  81 100 121 144 169 196 225]
 		 [256 289 324 361 400 441 484 529]]
 		的格式将矩阵表示为一个 字符串
 		!!! 注意返回值是字符串
		"""
		str_data = '[' + ''.join('[' + ''.join(f'\t{self.data[i][j]}' for j in range(self.dim[1])) + ']\n' for i in range(self.dim[0])) + ']'
		return str_data

	def det(self):
		r"""
		计算方阵的行列式。对于非方阵的情形应抛出异常。
		要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
		提示: Gauss消元
		
		Returns:
			一个 Python int 或者 float, 表示计算结果
		"""
		def is_upper_triangle(mat_data: list[list[int]]) -> bool:
			r"""
			用来判断输入矩阵是否为上三角矩阵,和while搭配使用
			
			Returns:
				True or False
			"""
			pass
		
		def row_exchange(mat_data: list[list[int]], r1: int, r2: int):
			r"""
			交换行列式中的两行,同时exchange记录交换次数

			Returns:
				exchange += 1
			"""
			pass
		
		def nonzero_search(mat_data: list[list[int]], column: int) -> int:
			r"""
			找到目标列非0的从上往下第一个行标号,如果没有,返回-1

			Returns:
				the index of the intended row
			"""
			pass
		"""
		实际上要实现原矩阵经过行初等变换成为阶梯型矩阵
		"""
		processing_data = self.data
		n = self.dim[0]
		exchange = 0
		row = 0

		while not (is_upper_triangle(processing_data) or row + 1 > n):
			nonzero_row = nonzero_search(processing_data)
			if not (nonzero_row + 1):
				return 0
			row_exchange(processing_data, nonzero_row, row)
			processing_data = [[processing_data[i][j] - (processing_data[i][0] / processing_data[row][0]) * processing_data[row][j] for j in range(n)] for i in range(row + 1, n + 1)]
			row += 1

		det_mat = (-1) ** exchange 
		for i in range(self.dim[0]):
			det_mat *= processing_data[i][i]
		return det_mat
		

	def inverse(self):
		r"""
		计算非奇异方阵的逆矩阵。对于非方阵或奇异阵的情形应抛出异常。
		要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
		提示: Gauss消元

		Returns:
			Matrix: 一个 Matrix 实例，表示逆矩阵
		"""
		pass

	def rank(self):
		r"""
		计算矩阵的秩
		要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
		提示: Gauss消元

		Returns:
			一个 Python int 表示计算结果
		"""
		pass

def I(n):
	'''
	return an n*n unit matrix
	'''
	return Matrix(data=[[1 for _ in range(n)] for _ in range(n)])

def narray(dim, init_value=1): # dim (,,,,,), init为矩阵元素初始值
	r"""
	返回一个matrix,维数为dim,初始值为init_value
	
	Args:
		dim: Tuple[int, int] 表示矩阵形状
		init_value: 表示初始值，默认值: 1

	Returns:
		Matrix: 一个 Matrix 类型的实例
	"""
	#return Matrix(dim, None, init_value)
	return Matrix(data=[[init_value for _ in range(dim[1])] for _ in range(dim[0])])

def arange(start, end, step):
	r"""
	返回一个1*n 的 narray 其中的元素类同 range(start, end, step)

	Args:
		start: 起始点(包含)
		end: 终止点(不包含)
		step: 步长

	Returns:
		Matrix: 一个 Matrix 实例
	"""
	return Matrix(data=[[i for i in range(start, end, step)]])

def zeros(dim):
	r"""
	返回一个维数为dim 的全0 narray

	Args:
		dim: Tuple[int, int] 表示矩阵形状

	Returns:
		Matrix: 一个 Matrix 类型的实例
	"""
	return narray(dim=dim, init_value=0)

def zeros_like(matrix: Matrix):
	r"""
	返回一个形状和matrix一样 的全0 narray

	Args:
		matrix: 一个 Matrix 实例
	
	Returns:
		Matrix: 一个 Matrix 类型的实例

	Examples:
		>>> A = Matrix(data=[[1, 2, 3], [2, 3, 4]])
		>>> zeros_like(A)
		>>> [[0 0 0]
			 [0 0 0]]
	"""
	return zeros(matrix.shape())

def ones(dim):
	r"""
	返回一个维数为dim 的全1 narray
	类同 zeros
	"""
	return narray(dim=dim)

def ones_like(matrix: Matrix):
	r"""
	返回一个维数和matrix一样 的全1 narray
	类同 zeros_like
	"""
	return narray(matrix.shape())

def nrandom(dim):
	r"""
	返回一个维数为dim 的随机 narray
	参数与返回值类型同 zeros
	"""
	return Matrix(data=[[random.randint(0, 100) for _ in range(dim[1])] for _ in range(dim[0])])

def nrandom_like(matrix: Matrix):
	r"""
	返回一个维数和matrix一样 的随机 narray
	参数与返回值类型同 zeros_like
	"""
	dim = matrix.shape()
	return Matrix(data=[[random.randint(0, 100) for _ in range(dim[1])] for _ in range(dim[0])])

def concatenate(items, axis=0):
	r"""
	将若干矩阵按照指定的方向拼接起来
	若给定的输入在形状上不对应，应抛出异常
	该函数应当不改变 items 中的元素

	Args:
		items: 一个可迭代的对象，其中的元素为 Matrix 类型。
		axis: 一个取值为 0 或 1 的整数，表示拼接方向，默认值 0.
			  0 表示在第0维即行上进行拼接
			  1 表示在第1维即列上进行拼接
	
	Returns:
		Matrix: 一个 Matrix 类型的拼接结果

	Examples:
		>>> A, B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])
		>>> concatenate((A, B))
		>>> [[0 1 2]
			 [3 4 5]]
		>>> concatenate((A, B, A), axis=1)
		>>> [[0 1 2 3 4 5 0 1 2]]
	"""
	if axis == 0:
		for i in range(len(items) - 1):
			if items[i].dim[1] != items[i + 1].dim[1]: raise ValueError
		return Matrix(
			data=[item.data[i] for item in items for i in range(item.dim[0])]
		)
	else:
		for j in range(len(items) - 1):
			if items[j].dim[0] != items[j + 1].dim[0]: raise ValueError
		return Matrix(
			data=[[items[k].data[m][n] for k in range(len(items)) for n in range(items[k].dim[1])] for m in range(items[0].dim[0])]
			)
	
def vectorize(func):
	r"""
	将给定函数进行向量化
	
	Args:
		func: 一个Python函数
	
	Returns:
		一个向量化的函数 F: Matrix -> Matrix, 它的参数是一个 Matrix 实例 x, 返回值也是一个 Matrix 实例；
		它将函数 func 作用在 参数 x 的每一个元素上
	
	Examples:
		>>> def func(x):
				return x ** 2
		>>> F = vectorize(func)
		>>> x = Matrix([[1, 2, 3],[2, 3, 1]])
		>>> F(x)
		>>> [[1 4 9]
			 [4 9 1]]
		>>> 
		>>> @vectorize
		>>> def my_abs(x):
				if x < 0:
					return -x
				else:
					return x
		>>> y = Matrix([[-1, 1], [2, -2]])
		>>> my_abs(y)
		>>> [[1, 1]
			 [2, 2]]
	"""
	def inner(matrix: Matrix):
		newdata = [[func(matrix.data[i][j]) for j in range(matrix.dim[1])] for i in range(matrix.dim[0])]
		return Matrix(data=newdata)
	return inner



if __name__ == "__main__":
	print("test here")
	pass
