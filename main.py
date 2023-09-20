import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



# print("Hello World!")
# array = np.zeros((5,3))
# print(array)
# array = np.random.random((2,4))
# print(array)


import seaborn as sns
import matplotlib.pyplot as plt

height = [62, 64, 69, 75, 66, 68, 65, 71, 76, 73]
weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187]
sns.scatterplot(x=height, y=weight)
plt.show()

gender = ["Female","Female","Female","Female","Male","Male","Male","Male"]
sns.countplot(gender)
plt.show()




