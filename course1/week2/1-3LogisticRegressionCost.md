# 1.3: Logistic Regression Cost Function

## Review So Far
- $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$, where $\sigma(z^{(i)}) = \frac{1}{1+ e^{-z^{(i)}}}$
- Given a training set $\{(x^{(1)}, y^{(1)}), ... , (x^{(m)}, y^{(m)})\}$,
want $\hat{y}^{(i)} \approx y^{(i)}$

## Loss Function/Error Function
- Could use $\mathcal{L}(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$, but the optimization problem is non-convex for logistic regression, making it susceptible to local optima
- Instead we use:
$$ \mathcal{L}(\hat{y}, y) = - \big(y \log{\hat{y}} + (1-y)\log{(1-\hat{y})}\big) $$
- Why does this function work well?
  - First of all, it's convex for logistic regression
  - If $y=1$, then $\mathcal{L}(\hat{y}, y) = -\log{\hat{y}}$
    - This means when we minimize $\mathcal{L}$, we want to make $\hat{y}$ large
  - If $y=0$, then $\mathcal{L}(\hat{y}, y) = -\log{(1-\hat{y})}$
    - This means when we minimize $\mathcal{L}$, we want to make $\hat{y}$ small

## Cost Function
$$ J(w,b) = -\frac{1}{m} \sum_{i=1}^{m}{\mathcal{L}(\hat{y}, y)} $$


## What's the difference?
- The *loss function* is the error of a single training example
- The *cost function* is the cost of your parameters
