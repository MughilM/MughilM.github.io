---
layout: post
title: "#76 - Counting summations"
date: 2017-06-21, 12:35
number: 76
tags: [10_diff]
---
> It is possible to write five as a sum in exactly six different ways:
> 
> $$
> \begin{aligned}
> &4 + 1
> \\
> &3 + 2
> \\
> &3 + 1 + 1
> \\
> &2 + 2 + 1
> \\
> &2 + 1 + 1 + 1
> \\
> &1 + 1 + 1 + 1 + 1
> \end{aligned}
> $$
> 
> How many different ways can one hundred be written as a sum of at least two positive integers?
{:.lead}
* * *

Breaking up a number into sets of sums like this are called finding the **partitions** of a number. It is an area of extensive research.

The traditional count of partitions of a number also include the number itself, and it is denoted $p(n)$. When presenting the final answer, we need to subtract one.

Intuitively, the value of $p(n)$ should somehow depend on the values before it (you can add one to all the partitions of $n-1$ and get $n$). [This article](https://en.wikipedia.org/wiki/Partition_function_(number_theory)#Recurrence_relations) shows the following recursive definition exists for $p(n$):

$$
p(n) = \sum_{k\neq 0}^\infty (-1)^{k+1}p\left(n - \frac{k(3k-1)}{2}\right)
$$
where $p(0)=1$ and $p(n)=0$ if $n<0$. Thus, this sum will contain finitely many non-zero terms. Additionally, $\frac{k(3k-1)}{2}$ steadily increases as $k$ goes $1, -1, 2, -2, 3, -3, 4, \dots$. In fact, these are pentagonal numbers. Therefore, using dynamic programming, the solution is simple to code, as we keep an array of the past values.
```python
# file: "problem076.py"
limit = 100

p = [0] * (limit + 1)
# Base case
p[0] = 1
for i in range(1, len(p)):
    k = 1
    # lambda function of pentagonal number
    pent = lambda x: x * (3 * x - 1) // 2
    while pent(k) <= i:
        p[i] += p[i - pent(k)] * int((-1) ** (k + 1))
        # If k is positive, then it turns into
        # its negative counterpart,
        # Otherwise, it goes to the next number
        if k > 0:
            k *= -1
        else:
            k = k * -1 + 1
print(p[limit] - 1)
```
The output after running is,
```
190569291
0.0013518999999999615 seconds.
```
Therefore, 100 can be written as a sum of positive integers in **190569291** ways.