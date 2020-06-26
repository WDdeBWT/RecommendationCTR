def func_1(x):
    x[0] += 1
    return x

li = [func_1, func_1, func_1]
a = [1]
b = a
for f in li:
    b = f(b)
print(a)
print(b)