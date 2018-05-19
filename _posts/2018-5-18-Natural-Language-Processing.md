---
layout: post
title: Natural Language Processing
---

A short introduction to natural language processing using scikit-learn's LDA package.

![_config.yml]({{ site.baseurl }}/images/NLP.png)


# Introduction

"Challenges in natural-language processing frequently involve speech recognition, natural-language understanding, and natural-language generation." - Wiki

## 'Optimal client recommendation for market makers in illiquid financial products'
"Given a historical record of corporate bond transactions and bond meta-data, we use a topic-modelling analogy to develop a probabilistic technique for compiling a curated list of client recommendations for a particular bond that needs to be traded" - Hendricks 2017 (https://arxiv.org/abs/1704.08488)


```python
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
```

```python
twenty_train = fetch_20newsgroups(shuffle=True, random_state=42, remove=("headers", "footers", "quotes"), 
                                  categories=['rec.autos',
                                            'rec.motorcycles',
                                             'rec.sport.baseball',
                                             'rec.sport.hockey'])
```

    Downloading 20news dataset. This may take a few minutes.
    Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)



```python
twenty_train.data
```

