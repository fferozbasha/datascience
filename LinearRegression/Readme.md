

# Linear Regression without Sklearn Library

The notebook is intended to perform Linear regression on a data set without using Sklearn. 
Objective of this implementation is to try and understand the internal logics of how the linear regression works. 

## Key underlying concepts

### Linear regression Formula

The formula for a linear regression is as below
```
y = wx + b
```

Where,
y = output variable
w = coefficient (weight) of the input variable
b = intercept (slope)

So, all that the linear regression model needs is to determine the best fit coefficient (w) and intercept (b).

### Cost Function

While trying to fit a line for the linear regression, it is nearly impossible to have the line predicting all the values exactly. There shall be a difference always between the predicted and the actual value. This difference is referred as the cost (error) for each prediction. 

So, while trying to identify the best fit line, the intention will be to find the line which has the least possible error (cost).

```
Cost of each prediction = ( Predicted Value - Actual value ) ** 2
Total cost of the input data = Sum of cost of all predictions / total number of input data
```
### Gradient Descent

If we manually try generating multiple lines with different possible values for w and b, and try to plot a chart with w, b and the total cost, we might see, that at one point the cost will be the least. 

Challenge is to find that w and b, where the cost function is the least. 

Gradient descent is the algorithm that helps in determining the same. The way it works is, 

* Start the algorithm with random value for w & b.
* Calculate the cost for that combination of values. 
* Then, using the derivative approach, shall try to find the next best possible value for w and b. 
* At each iteration, the cost function shall reduce.
* Continue iteration, until the cost function converges (or has reached the best minimum value).

Output of the above process will be the best possible values for w and b.

With these, the missing pieces for the linear regression formula will be completed. 
```
y = wx + b
```


