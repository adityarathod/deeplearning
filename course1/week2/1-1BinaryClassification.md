**WEEK 2, SECTION 1: LOGISTIC REGRESSION AS A NEURAL NETWORK**
# 1.1: Binary Classification

- You might want to use a for loop, but vectorization is better
- A neural network has two steps:
  - Forward pass/propagation
  - Backward pass/propagation
- The concepts will be explained in the concept of Logistic Regression

## What is Logistic Regression?
- It's a binary classification problem
  - "Is a picture a cat (1), or not a cat (0)?"
- The picture is represented to a computer as a matrix of RGB intensity values
  - If image is 64x64px, then you have 3 64x64 matrices, each corresponding to R,G,B channels
- To feed this into a NN, we need to "unroll" matrices into an input feature vector $x$:
  $$ x= \begin{bmatrix} 255 \\ 231 \\ ... \\ ... \\ 255 \\ 134 \\ ... \\ ... \end{bmatrix}$$
- The dimensionality of the vector $x$ is 64x64x3, or `12288`, meaning $x \in \mathbb{R}^{12288}$. Thus, the number of features $n_x = 12288$ (sometimes abbreviated as just $n$)


## Notation For the Course
- A training example looks like this:
$$ (x,y)\text{, where } x \in \mathbb{R}^{n_x}\text{ and } y \in \{0,1\}$$
- There are $m$ training examples in our dataset, and are written like this:
$$ \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}),...,(x^{(m)}, y^{(m)})\} $$
- Usually just $m$ refers to $m_{train}$, the number of examples in the training set, while $m_{test}$ is the number of examples in our test set
- A more compact notation for our training set looks like this (the training example vectors are stacked as columns and the dimensions are $n_x$ rows and $m$ columns):
$$ X = \begin{bmatrix} x^{(1)}, \ x^{(2)}, \ ..., \ x^{(m)} \end{bmatrix} $$  
$$ X \in \mathbb{R}^{n_x \times m}$$
- The dimensions of $X$ are expressed like this in Python:
  - `X.shape = (n_x, m)`
- Also, the matrix $Y$ stores our training set outputs ($Y \in \mathbb{R}^{1\times m}$):
$$ Y = \begin{bmatrix} y^{(1)}, \ y^{(2)}, \ ..., \ y^{(m)} \end{bmatrix} $$
- The dimensions of $Y$ are expressed like this in Python:
  - `X.shape = (1, m)`
