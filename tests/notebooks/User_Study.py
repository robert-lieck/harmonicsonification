#!/usr/bin/env python
# coding: utf-8

# # User Study

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import harmonicsonification as hs
hs.seed_everything(42)


# ## Data Set Creation

# We want to test whether our harmonic sonification approach allows users to distinguish typical data points from outliers just based on the sound. For this, we first create an artificial data set with the characteristics typically obtained from applying PCA (i.e. dimensions with decreasing variance). We then create an _outlier_ data set with uniformly distributed points (some of these points will be like typical data points, but most will fall outside the distribution).

# In[2]:


dim = 16                               # dimensions
n_data = 100                           # number of data points
std = np.exp(-np.linspace(0, 4, dim))  # standard deviations
uniform = 3                            # width of the uniform outlier distribution

columns = [f"dim {i + 1}" for i in range(dim)]
data = pd.DataFrame(np.random.normal(size=(n_data, dim)) * std, columns=columns)
outlier = pd.DataFrame(np.random.uniform(-uniform, uniform, size=(n_data, dim)), columns=columns)
data['type'] = 'data'
outlier['type'] = 'outlier'
print(f"{dim} dimensions")
print(f"{n_data} data points")
print(f"std: {std.round(2)}")
print(f"outliers in [-{uniform}, {uniform}]")
plt.plot(std, '-o');


# We can create scatter plots for all pairs of dimensions, which gives a rough idea of the two distributions.

# In[3]:


# combined = pd.concat([outlier, data], ignore_index=True)
# p = sns.pairplot(combined, hue='type', palette=['blue', 'red'], diag_kind=None)
# for ax in p.axes.flatten():
#     ax.set_xlim(-uniform, uniform)
#     ax.set_ylim(-uniform, uniform)


# ## Sonification

# Lowest and highest frequency that may appear in the sonification, just for reference.

# In[4]:


base_freq = 110
amps = np.zeros(dim)
amps[-1] = 1
hs.sonify_am(x=amps, f0=base_freq).display()


# Helper functions to extract data points, randomise their order (for the experiment), and sonify them

# In[5]:


def get_points(data, outlier, shuffle):
    points = []
    for d, l in [(data, 'data'), (outlier, 'outlier')]:
        if d is not None:
            d = d[:,:-1]
            points += [(d, f'{l} {i + 1}') for i, d in enumerate(d)]
    if shuffle:
        random.shuffle(points)
    points, labels = list(zip(*points))
    return np.array(points), labels


# In[6]:


def sonify(points, labels, std, base_freq, add_fundamental=True, label=False, print_amps=False):
    points = np.abs(points)
    points /= std[None, :]
    points /= points.max()
    for i, (p, l) in enumerate(zip(points, labels)):
        if add_fundamental:
            amps = [1] + list(p)
        else:
            amps = p
        amps = np.array(amps, dtype=float)
        if label:
            print(l)
        else:
            print(f"point {i + 1}")
        if print_amps:
            print(amps.round(2))
        hs.sonify_am(x=amps, f0=base_freq).display()


# ### Example Data

# Here are some typical data points as well as some outliers.

# In[7]:


n_examples = 5
points, labels = get_points(data.values[:n_examples], outlier.values[:n_examples], False)


# In[8]:


sonify(points, labels, std, base_freq, label=True)


# ### Trials

# Now we get some data points and outliers shuffle them randomly and let participants guess.

# In[9]:


n_test = 10
points, labels = get_points(data=data.values[n_examples:n_examples+n_test], 
                            outlier=outlier.values[n_examples:n_examples+n_test], 
                            shuffle=True)


# For the evaluation, we once print their correct labels.

# In[10]:


for l in labels:
    print(l)
print("--------------------")
sonify(points, labels, std, base_freq, label=True)


# Now we just print a point index (this is shown to the participants)

# In[11]:


sonify(points, labels, std, base_freq)

