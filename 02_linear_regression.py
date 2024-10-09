import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd

data = pd.read_csv("Salary_Data.csv")
#plt.scatter(data.YearsExperience, data.Salary)
#plt.show()

m = 0
b = 0
L = 0.01

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y - (m * x + b)) ** 2
        total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) *  (y - (m_now * x + b_now))
    
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

epochs = 2000

for i in range(epochs):
    if i % 50 == 0:
        print(f"epoch : {i}")
    m, b = gradient_descent(m, b, data, L)

print(m, b)
plt.scatter(data.YearsExperience, data.Salary, color="black")
plt.plot(list(range(0, 14)), [m * x + b for x in range(0, 14)], color="red")
plt.show()


print(data)
