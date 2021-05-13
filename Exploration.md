# Project Exploration

## Tasks Completed
After using all watch brands as features at once for my data, I decided to seperate the data
by watch brands one at a time, which decreased the amount of empty data I would be looping through. I also decided to explore Activity this time around. Given that this was categorical data, I first had to transform it into numerical data and created three separate features (removed 
data points where activity was NaN) This meant I was now only looking at 6 features at a time: 

1. Watch Brand (Fitbit or Apple) Guess
2. Skin Tone
3. Activity (Seperated into : Rest, Breathe, Activity, Type)

The reason I chose to look at the smart watch data seperately was mainly due to the fact that one brand had no correlation to the other's performance and also there were many 0s because only the heartbeat for 1 watch brand was taken at a time. I have also chosen to explore RandomTrees and DecisionTrees to see the performance of each model.


## Results
Here are results without activity category:

```python

# Including Apple Only
'''
Dtree Min Error Score =  ExperimentResult(vali_acc=-0.004738203521941431, params={'criterion': 'poisson', 'max_depth': 1, 'random_state': 0}, model=DecisionTreeRegressor(criterion='poisson', max_depth=1, random_state=0))

RTree Min Error Score =  ExperimentResult(vali_acc=0.4762177692270869, params={'criterion': 'mae', 'max_depth': 1, 'random_state': 0}, model=RandomForestRegressor(criterion='mae', max_depth=1, random_state=0))
'''

# Including FitBit Only
'''
Best DTree ExperimentResult(vali_acc=0.0035854097419107944, params={'criterion': 'poisson', 'max_depth': 1, 'random_state': 1}, model=DecisionTreeRegressor(criterion='poisson', max_depth=1, random_state=1))

Best RForest ExperimentResult(vali_acc=0.1797829030631689, params={'criterion': 'mae', 'max_depth': 1, 'random_state': 0}, model=RandomForestRegressor(criterion='mae', max_depth=1, random_state=0))
'''

```


Here are the results with activity categories setup as numerical data:
```python
# Including Apple Only
'''
Best DTree ExperimentResult(vali_acc=0.043089133607391084, params={'criterion': 'poisson', 'max_depth': 1, 'random_state': 0}, model=DecisionTreeRegressor(criterion='poisson', max_depth=1, random_state=0))

Best RForest ExperimentResult(vali_acc=0.5360062617847372, params={'criterion': 'mae', 'max_depth': 1, 'random_state': 2}, model=RandomForestRegressor(criterion='mae', max_depth=1, random_state=2))

Best SGD ExperimentResult(vali_acc=-2.735139523020645e+22, params={'random_state': 2, 'penalty': 'elasticnet', 'max_iter': 100, 'loss': 'squared_loss'}, model=SGDRegressor(max_iter=100, penalty='elasticnet', random_state=2))
'''

# Including FitBit Only
'''
Best DTree ExperimentResult(vali_acc=0.03009721690768108, params={'criterion': 'poisson', 'max_depth': 1, 'random_state': 0}, model=DecisionTreeRegressor(criterion='poisson', max_depth=1, random_state=0))

Best RForest ExperimentResult(vali_acc=0.33027962672278766, params={'criterion': 'mse', 'max_depth': 1, 'random_state': 0}, model=RandomForestRegressor(max_depth=1, random_state=0))

Best SGD ExperimentResult(vali_acc=-1.4000078859172049e+22, params={'random_state': 1, 'penalty': 'elasticnet', 'max_iter': 100, 'loss': 'squared_loss'}, model=SGDRegressor(max_iter=100, penalty='elasticnet', random_state=1))
'''

```

## Next Steps
I hope to explore other watch brands now that I have decided to seperate my data by one watch
brand at a time. I also hope to combine two models together (Boosting) for my next steps and see
the results of them working together as well as practice weighing features. I would also like to create more visuals and interactive examples to show how these models are working rather than just their performance, specifically to help with the next deliverable which includes some kind of presentation. 

While previously testing separate models, I understand that some non-linear models take a lot more
time to run and would be ineffective in using in the real world so I have chosen to not try and 
look further into some models that are more computationally heavy (MLP). Granted, my computer also cannot handle it and will shut down, but that just means it might also be inneficient for others.


## Summary
Overall, I think the performance of my non-linear models are really great. I had a very low error score.
Despite this, I would like to make sure I am not overfitting so I need to explore other methods
to understand the methods better. There are other methods of modifying my models and so
for the next delivarable I will be limiting myself to the tasks I mentioned in my next steps.
