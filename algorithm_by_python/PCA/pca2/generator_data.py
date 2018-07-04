# 来自http://www.hankcs.com/ml/python-step-by-step-to-achieve-the-principal-component-analysis.html
# 数据的生成代码如下
import numpy as np

np.random.seed(23434)

mu_vec1 = np.array([0, 0, 0])
cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# np.random.multivariate_normal方法用于根据实际情况生成一个多元正态分布矩阵
# multivariate_normal(mean, cov, size=None, check_valid=None, tol=None)
# 其中mean和cov为必要的传参而size，check_valid以及tol为可选参数：
# mean：mean是多维分布的均值(每一个维度)；
# cov：协方差矩阵，注意：协方差矩阵必须是对称的且需为半正定矩阵；
# size：指定生成的正太分布矩阵的维度（例：若size=(1, 1, 2)，则输出的矩阵的shape即形状为 1X1X2XN（N为mean的长度））。
# check_valid：这个参数用于决定当cov即协方差矩阵不是半正定矩阵时程序的处理方式，它一共有三个值：warn，raise以及ignore。
# 当使用warn作为传入的参数时，如果cov不是半正定的程序会输出警告但仍旧会得到结果；当使用raise作为传入的参数时，如果cov不是
# 半正定的程序会报错且不会计算出结果；当使用ignore时忽略这个问题即无论cov是否为半正定的都会计算出结果。
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3, 20)

mu_vec2 = np.array([1, 1, 1])
cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class1_sample.shape == (3, 20)