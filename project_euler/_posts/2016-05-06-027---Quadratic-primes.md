---
layout: post
title: "#27 - Quadratic primes"
date: 2016-05-06 20:01
number: 27
tags: [05_diff]
---
> Euler discovered the remarkable quadratic formula:
> 
> $$
> n^2+n+41
> $$
> 
> It turns out that the formula will produce 40 primes for the consecutive integer values $0\leq n\leq 39$. However, when $n = 40, 40^2+40+41=40(40+1)+41$ is divisible by 41, and certainly when $n=41,41^2+41+41$ is clearly divisible by 41.
> 
> The incredible formula $n^2-79n+1601$ was discovered, which produces 80 primes for the consecutive values $0\leq n\leq 79$. The product of the coefficients, -79 and 1601, is -126479.
> 
> Considering quadratics of the form:
> 
> $$
> n^2+an+b,\text{ where } |a|<1000\text{ and } |b|\leq 1000
> $$
> 
> where the bars indicate absolute value.
> 
> Find the product of the coefficients, $a$ and $b$, for the quadratic expression that produces the maximum number of primes for consecutive values of $n$, starting $n=0$.
{:.lead}
* * *

Initially, it feels we need to loop through all values of $a$ and $b$. However, there are a couple of observations.

Let $f(n) = n^2+an+b$. The problem states $f(0)$ must be prime. However, $f(0) = b$, which means $\mathbf{b}$ **must be prime**. There are about 150 primes below 1000, so that immediately reduces our search space. Additionally, $b\neq 2$, because if $n$ is even, then $n^2+an + 2$ will be even, and thus won't be prime.

In addition to saying $b$ is prime (and $b\neq 2$), we can also say that $a$ is odd. Observe what happens when $a$ is even and $n$ is odd:

$$
\begin{aligned}
f(Odd) &= Odd^2 + Even\times Odd+Odd \\
&= Odd + Even + Odd
\\ &=
Odd + Odd \\
&= Even
\end{aligned}
$$

Every other $n$, $f(n)$ will be even, and hence be composite. 

Finally, $a > -b$, because otherwise, that allows $f(n)$ to be negative. In the end, we need a double for loop that checks our constrained search space. For testing if $f(n)$ is prime, we can do the generic loop until $\sqrt{n}$.
```python
# file: "problem027.py"
def isPrime(p):
    if p <= 1:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    for i in range(3, int(p ** 0.5) + 1):
        if p % i == 0:
            return False
    return True

limit = 1000
Bs = primesieve.primes(limit)[1:]
maxChain = 40
maxA = 1
maxB = 41
for b in Bs:
    for a in range(-b + 2, limit):
        # Make the function
        f = lambda x: x ** 2 + a * x + b
        # We know f(0) = b is prime, so start from n = 1.
        n = 1
        while isPrime(f(n)):
            n += 1
        # Check to see if chain length is bigger...
        if n > maxChain:
            maxChain = n
            maxA = a
            maxB = b

print(maxA, maxB, maxChain, maxA * maxB)
```
The result is,
```
-61 971 71 -59231
0.337677999981679 seconds.
```
Thus, $f(n) = n^2 - 61n + 971$ produces a chain of 71 consecutive primes from $0\leq n\leq 70$. The product we want is **-59231**.