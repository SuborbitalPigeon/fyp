import numpy as np

def file_to_matrix(filename):
    with open(filename) as f:
        data = f.readlines()

    data = [line.split() for line in data]
    return np.array(data[:3], dtype=float) # strip the last newline

if __name__ == '__main__':
    mat = file_to_matrix('bark/H1to2p')
    print(mat)
