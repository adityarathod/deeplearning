**WEEK 2, SECTION 2: PYTHON AND VECTORIZATION**

# 2.1: Vectorization
- Vectorization = the art of removing explicit for loops in code


## What is Vectorization?
- Let's say we have an equation $z = w^T x + b$, where $w, x \in \mathbb{R}^{n_x}$.
- There are two ways to compute this, non-vectorized and vectorized:

```python
# Non-vectorized
z = 0
for i in range(n_x):
  z += w[i] * x[i]
z += b
```

```python
# Vectorized
z = np.dot(w, x) + b
```

- Both CPUs and GPUs have special instructions, called SIMD instructions, that make your code faster
  - Using `np.dot` instead of explicit for loops makes your code faster for this reason
