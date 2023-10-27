import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tinh_gia_tri_lop_dau_vao(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return sigmoid(z)

# Nhập dữ liệu đầu vào
print("Nhập số lượng đầu vào: ", end='')
n1 = int(input())
in_list = [float(input("Nhập đầu vào {}: ".format(i + 1))) for i in range(n1)]

# Lớp ẩn
print("Nhập số lượng nơ-ron ẩn: ", end='')
n_hidden = int(input())

hidden_weights = np.array([[float(input("Nhập trọng số từ đầu vào {} đến nơ-ron ẩn {}: ".format(i + 1, j + 1))) for j in range(n_hidden)] for i in range(n1)])

hidden_biases = np.array([0.25] * n_hidden)
print("Hidden_weights: ",hidden_weights)

hidden_outputs = tinh_gia_tri_lop_dau_vao(in_list, hidden_weights, hidden_biases)
print("Kết quả đầu ra của lớp ẩn: ", hidden_outputs)
print(type(hidden_outputs))
# Lớp đầu ra
print("Nhập số lượng đầu ra: ", end='')
n_output = int(input())

outputs_weights = np.array([[float(input("Nhập trọng số từ nơ-ron ẩn {} đến đầu ra {}: ".format(i + 1, j + 1))) for j in range(n_output)] for i in range(n_hidden)])

outputs_biases = np.array([0.35] * n_output)
print("outputs_weights:\n",outputs_weights)
output = tinh_gia_tri_lop_dau_vao(hidden_outputs, outputs_weights, outputs_biases)
print("Kết quả đầu ra:\n", output)
