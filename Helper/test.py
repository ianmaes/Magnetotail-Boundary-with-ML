import numpy as np
import matplotlib.pyplot as plt

def func(a=np.arange(10), b=np.arange(0, 20, 2), **kwargs):
    plt.plot(a, b, **kwargs)
    plt.show()

if __name__ == "__main__":
    func(a=np.arange(10), b=np.arange(0, 20, 2), color='red', linestyle='--')