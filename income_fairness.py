# load dataset
from pycaret.datasets import get_data
income = get_data('income')

# init setup
from pycaret.classification import *
exp_name = setup(data = income,  target = 'income >50K')

# train model
lr = create_model('lr')

# check model fairness
lr_fairness = check_fairness(lr, sensitive_features = ['sex', 'race'])