# mean-reversion

logic：
Using six features as input, as well as label(whether the return of next day is larger than median/0.003), train the tree model and use it to predict the future.
long (predictive probability)-0.5 unit if probability >0.5 and short 0.5-(predictive probability) unit

classification.py:
A file contains feature generation,labeling,train test spliting, model training and model evaluation.

summary:
simple test to find the optimal hyperparameter for mean-reversion and momentum.
