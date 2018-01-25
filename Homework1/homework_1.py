import scipy.io as sp

data = sp.loadmat("./data_class4.mat")["Data"][0]

print(data[0])
for d in data[0]:
    print(d)
