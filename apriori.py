# -*- coding: utf-8 -*-


! pip install apyori

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Data Preprocessing"""

dataset=pd.read_csv('/content/Market_Basket_Optimisation.csv',header=None)
transactions =[]
for i in range(0,7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

"""## Training the Apriori model on the dataset"""

from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3.0, min_length=2, max_length=2)

"""## Visualising the results



results=list(rules)
results

"""### Putting the results well organised into a Pandas DataFrame"""

def inspect(results):
  lhs=[tuple(result[2][0][0])[0] for result in results]
  rhs=[tuple(result[2][0][1])[0] for result in results]
  supports=[result[1] for result in results]
  confidence=[result[2][0][2]for result in results]
  lifts=[result[2][0][3] for result in results]
  return list(zip(lhs,rhs,supports,confidence,lifts))
resultsinDataFrame=pd.DataFrame(inspect(results),columns=["LHS","RHS","Support","confidence","lift"])

resultsinDataFrame

"""### Displaying the results non sorted

### Displaying the results sorted by descending lifts
"""

resultsinDataFrame.nlargest(n=15,columns='lift')