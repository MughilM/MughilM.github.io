---
layout: post
title: "#77 - Prime summations"
date: 2018-06-05 12:18
number: 77
tags: [25_diff]
---
> It is possible to write ten as the sum of primes in exactly five different ways:
>
> $$
> \begin{aligned}
> &7+3
> \\
> &5+5
> \\
> &5+3+2
> \\
> &3+3+2+2
> \\
> &2+2+2+2+2
> \end{aligned}
> $$
>
> What is the first value which can be written as the sum of primes in over five thousand different ways?
{:.lead}
* * *

Like with [#76 - Counting summations](/blog/project_euler/2017-06-21-076-Counting-summations){:.heading.flip-title}, I researched for a possible recursive definition when using only prime numbers. [This Math StackExchange post](https://math.stackexchange.com/a/89661) shows a way to do it.

The number of partitions $\kappa(n)$ is given by

$$
\kappa(n) = \frac{1}{n}\left(\text{sopf}(n) + \sum_{j=1}^{n-1}\text{sopf}(n)\kappa(n-j)\right)
$$

where $\kappa(1) = 0$. The function $\text{sopf}(n)$ is the **sum of prime factors** function, where we add the **distinct** prime factors of $n$.

To find distinct prime factors, we can completely divide out a prime factor when we encounter a factor to save some time. The number of partitions generally explodes exponentially, so we do not need to do anything fancy with our prime search. The summation involves multiplying the reverse of the values, and I use `numpy` to quickly do element-wise multiplication.
```python
# file: "problem077.py"
def sopf(n):
    s = 0
    i = 2
    while i <= n:
        if n % i == 0:
            s += i
            # Completely divide out the factor
            while n % i == 0:
                n //= i
        else:
            i += 1
    return s

sopfs = [0, 0]
kappas = [0, 0]
n = 2
while kappas[-1] <= 5000:
    sopfs.append(sopf(n))
    kappas.append(1/n * (sopfs[n] + sum(sopfs[1:n] * np.array(kappas[1:][::-1]))))
    n += 1
print(len(kappas) - 1, 'is the first value which can be written in 5000+ ways.')
```
The extra 0 in the beginning is so that the index matches the value of $n$. Running this results in,
```
71 is the first value which can be written in 5000+ ways.
0.0018762999999999974 seconds.
```
Therefore, **71** is the first integer such that there are at least 5000 different ways to write sums.