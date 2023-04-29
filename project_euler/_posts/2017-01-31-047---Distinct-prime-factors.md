---
layout: post
title: "#47 - Distinct prime factors"
date: 2017-01-31 20:00
number: 47
tags: [05_diff]
---
> The first two consecutive numbers to have two distinct prime factors are:
> 
> $$
> \begin{aligned}
> 14 &= 2\times 7
> \\
> 15 &= 3\times 5
> \end{aligned}
> $$
>
> The first three consecutive numbers to have three distinct prime factors are:
> 
> $$
> \begin{aligned}
> 644 &= 2^2\times 7\times 23
> \\
> 645 &= 3\times 5\times 43
> \\
> 646 &= 2\times 17\times 19
> \end{aligned}
> $$
> 
> Find the first four consecutive integers to have four distinct prime factors each. What is the first of these numbers?
{:.lead}
* * *

We can write a function to quickly compute distinct prime factors, by completely dividing out a factor before moving to the next one.

We can also do some optimizations on the search. For example, let's say a candidate set $n$, $n+1$, $n+2$, and $n+3$, has 4, 4, 4, and 3 prime factors respectively. This $n$ is clearly not a candidate. However, we don't need to check again starting from $n+1$, since we know $n+3$ has only 3 factors. The next number we check is $n+4$. In this way, we can skip a lot of numbers.
```python
# file: "problem047.py"
# Start from 647...
primes = primesieve.primes(500000)
n = 647
while True:
    # Calculate starting from this number
    # the number of distinct prime factors.
    count = 0
    while len(set(distinctPrimeFacts(n, primes))) == 4:
        count += 1
        n += 1
    if count == 4:
        for i in range(n - 4, n):
            print(i, '==>', distinctPrimeFacts(i, primes))
        break
    n += 1
```
The output is,
```
134043 ==> [3, 7, 13, 491]
134044 ==> [2, 2, 23, 31, 47]
134045 ==> [5, 17, 19, 83]
134046 ==> [2, 3, 3, 11, 677]
1.6933869983913123 seconds.
```
Therefore, the set of 4 integers we are looking for are,

$$
\begin{aligned}
134043 &= 3\times 7\times 13\times 491
\\
134044 &= 2^2\times 23\times 31\times 47
\\
134045 &= 5\times 17\times 19\times 83
\\
134046 &= 2\times 3^2\times 11\times 677
\end{aligned}
$$
and our starting integer is **134043**.