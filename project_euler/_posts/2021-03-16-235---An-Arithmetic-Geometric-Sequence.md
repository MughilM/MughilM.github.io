---
layout: post
title: "#235 - An Arithmetic Geometric Sequence"
date: 2021-03-16 15:56
number: 235
tags: [40_diff]
---
>Given the arithmetic-geometric sequence $u(k) = (900-3k)r^{k-1}$.
>Let $s(n)=\sum_{k=1}^n u(k)$.
>
>Find the value of $r$ for which $s(5000) = -600\,000\,000\,000$.
>
>Give your answer rounded to 12 places behind the decimal point.
* * *

## Introduction
We have a few different variables here, but the problem statement essentially tells us to solve for $r$ in the following equation:

$$
\sum_{k=1}^{5000}(900-3k)r^{k-1} = -600\,000\,000\,000
$$

Most of this problem will be spent finding a direct formula for the summation. To make things simpler and a little bit more concise, I'll revert 5000 back to $n$. We can split up the sum as follows:

$$
\sum_{k=1}^n(900-3k)r^{k-1} = 900\sum_{k=1}^n r^{k-1} - 3\sum_{k=1}^n kr^{k-1}
$$

## First Summation
Recall the sum of a finite geometric series is $\sum_{i=0}^n a^i = \frac{a^{n+1}-1}{a-1}$. The first summation is extremely similar to this, and so we can apply the formula directly:

$$
\sum_{k=1}^n r^{k-1} = \sum_{k=0}^{n-1} r^k = \frac{r^{n-1+1}-1}{r-1} = \frac{r^n - 1}{r - 1}
$$

## Second summation
This sum is much trickier, because of the $k$ in front of the power. But we can still compute this analytically. First, let our desired sum $T(n) = \sum_{k=1}^n kr^{k-1}$. Let's expand this sum, and multiply both sides by $r$ and $r^2$ and observe what happens.

$$
\begin{aligned}
T(n) &= 1 + 2r + 3r^2 + 4r^3 + \cdots + (n-1)r^{n-2} + nr^{n-1}
\\
rT(n) &= r + 2r^2 + 3r^3 + 4r^4 + \cdots + (n-1)r^{n-1} + nr^n
\\
r^2 T(n) &= r^2 + 2r^3 + 3r^4 + \cdots + (n-2)r^{n-1} + (n-1)r^n + nr^{n+1}
\end{aligned}
$$

If we subtract the third equation from the second, we observe that each coefficient of $r^i$ (except for $r^{n+1}$) reduce to 1.

$$
\left(r - r^2\right)T(n) = \left(r + r^2 + r^3 + \cdots + r^{n-1} + r^n \right) - nr^{n+1}
$$

The expression in the parentheses is simply the regular sum of a finite geometric series $\sum_{i=1}^n r^i = \frac{r^{n+1} - 1}{r-1} - 1$. We can substitute this in and solve for $T(n)$:

$$
\begin{aligned}
\left(r - r^2\right)T(n) &= \frac{r^{n+1} - 1}{r-1} - 1 - nr^{n+1}
\\
T(n) &= \frac{1}{r(1-r)}\left( \frac{r^{n+1} - 1}{r-1} - \frac{r-1}{r-1} - \frac{n(r-1)r^{n+1}}{r-1} \right)
\\ &= -\frac{1}{r(r-1)^2}\left( r^{n+1} - 1 - r + 1 - n(r - 1)r^{n+1} \right)
\\ &= -\frac{1}{r(r-1)^2}\left( r^{n+1}(1-nr+n) - r \right)
\\ &= \frac{1}{(r-1)^2}\left( 1 - r^n(1-nr+n) \right)
\end{aligned}
$$

## Putting it together
We have both pieces, now we can get a direct formula for $s(n)$:

$$
\begin{aligned}
s(n) &= 900\sum_{k=1}^n r^{k-1} - 3\sum_{k=1}^n kr^{k-1}
\\ &= 900\left( \frac{r^n-1}{r-1} \right) - \frac{3}{(r-1)^2}\left( 1 - r^n(1-nr+n) \right)
\\ &= \frac{1}{(r-1)^2}\left( 900(r^n-1)(r-1) + 3r^n(1-nr+n) - 3 \right)
\\ &= \frac{1}{(r-1)^2}\left( -3nr^{n+1}+900r^{n+1} + 3nr^n - 897r^n - 900r + 897 \right)
\\ &= -\frac{3}{(r-1)^2}\left( r^{n+1}(n-300) + r^n(299-n) + 300r - 299 \right)
\end{aligned}
$$

This means we eventually have to solve the following equation for $r$. Let $s(5000)=h(r)$:

$$
h(r)=\frac{3}{(r-1)^2}\left( 4700r^{5001} - 4701r^{5000} + 300r - 299 \right) = 6\times 10^{11}
$$

## Solving the equation
How do we solve this equation? The powers on $r$ are extremely large, and because that, it's possible that $h(r)$ blows up very quickly to the exponent. There are many numerical methods out there to solve high degree polynomials, but we must careful not to introduce too many floating point errors, as our desired solution must be correct to 12 decimal places.

We can utilize **Newton's Method**, which utilizes the derivative, to solve this. This method is used to numerically find the zeroes of a function $f(x)$. Assuming that $f$ is differentiable, and starting with an initial guess of $x_0$, then the next guess $x_{i+1}$ is

$$
x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}
$$

This also can easily be adapted to find _any_ specific value of $f$, rather than 0. For example, if we want to find $x$ such that $f(x) = c$, this is the same as finding the zero of the function $f(x)-c$. Therefore, we will apply Newton's Method to the function $f(r)= h(r) - 6\times 10^{11}$. This function is differentiable everywhere, except at $r=1$. We need to be careful to make sure our guesses do not land exactly at 1. We should be safe as long as our initial guess is not 1.
### Computing the derivative
We will make this slightly easier on ourselves. If we let $g(r) = 4700r^{5001} - 4701r^{5000} + 300r - 299$, then we have 

$$
f(r) = \frac{3g(r)}{(r-1)^2} - 6\times 10^{11}
$$

and using the quotient rule, its derivative $f'(r)$ is

$$
\begin{aligned}
	f'(r) &= \frac{3(r-1)^2 g'(r) - 6(r-1)g(r) }{(r - 1)^4}
	\\ &=
    \frac{3(r-1)g'(r) - 6g(r)}{(r-1)^3}
\end{aligned}
$$
### Determining an initial guess
Even though Newton's Method is powerful, a "wrong" initial guess can lead to unstable results and not converge to the answer we want. Thus, it would be very helpful if our initial guess was still somewhat close to a 0. For our function $f$, it is undefined at $r=1$, and $f(2) >> 6\times 10^{11}$. Please note that $f$ is also exhibits extreme exponential behaviour. This means that our answer is actually very close to 1, and so let's set our initial guess at $r=1.01$.
## Solution
With the function definitions we have, it is fairly straightforward to implement Newton's Method in code. We create separate methods for $f$, $g$, $f'$, and $g'$ to improve readability. Since we need to be correct to at least 12 decimal places, we run the loop until $\left| f(r_{i+1}) - f(r_i)\right| < 10^{-12}$.

```python
def g(r):
    return 4700 * r ** 5001 - 4701 * r ** 5000 + 300 * r - 299

def g_derivative(r):
    return 23504700 * r ** 5000 - 23505000 * r ** 4999 + 300

def f(r):
    return 3 * g(r) / ((r - 1) ** 2) - 6e11

def f_derivative(r):
    return (3 * (r - 1) * g_derivative(r) - 6 * g(r)) / ((r - 1) ** 3)

curr_guess = 1.01
new_guess = curr_guess - f(curr_guess) / f_derivative(curr_guess)

while math.fabs(new_guess - curr_guess) > 1e-12:
    curr_guess = new_guess
    new_guess = curr_guess - f(curr_guess) / f_derivative(curr_guess)

print(f'{new_guess:.12f}')
```
Running this short loop we have
```bash
1.002322108633
0.0002351950006413972 seconds.
```

Therefore, the value of $r$ which solves the problem is **1.002322108633**.


