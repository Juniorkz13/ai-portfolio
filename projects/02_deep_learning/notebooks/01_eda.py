#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# In[2]:


df = pd.read_csv(
    "../data/raw/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"]
)

df.head()


# In[3]:


sns.countplot(x="label", data=df)
plt.title("Distribuição das Classes (Spam vs Ham)")
plt.show()

df["label"].value_counts(normalize=True)


# In[4]:


df["text_length"] = df["text"].apply(len)

sns.histplot(df["text_length"], bins=50)
plt.title("Distribuição do Tamanho dos Textos")
plt.show()

