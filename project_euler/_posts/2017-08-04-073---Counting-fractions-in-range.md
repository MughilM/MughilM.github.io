---
layout: post
title: "#73 - Counting fractions in a range"
date: 2017-08-04 14:52
number: 73
tags: [15_diff]
---
> Consider the fraction, $n/d$, where $n$ and $d$ are positive integers. If $n<d$ and $HCF(n,d)=1$, it is called a reduced proper fraction.
> 
> If we list the set of reduced proper fractions for $d\leq 8$ in ascending order of size, we get:
> 
> $$
> \frac{1}{8},\frac{1}{7},\frac{1}{6},\frac{1}{5},\frac{1}{4},\frac{2}{7},\frac{1}{3},\mathbf{\frac{3}{8}},\mathbf{\frac{2}{5}}, \mathbf{\frac{3}{7}},\frac{1}{2},\frac{4}{7},\frac{3}{5},\frac{5}{8},\frac{2}{3},\frac{5}{7},\frac{3}{4},\frac{4}{5},\frac{5}{6},\frac{6}{7},\frac{7}{8}
> $$
> 
> It can be seen that there are 3 fractions between $1/3$ and $1/2$.
> 
> How many fractions lie between $1/3$ and $1/2$ in the sorted set of reduced proper fractions for $d\leq12000$?
{:.lead}
* * *

The "set of reduced proper fractions" has a name called the [**Farey sequence**](https://en.wikipedia.org/wiki/Farey_sequence). The link also shows how to generate the fractions **in order** given two consecutive fractions in the sequence.

Paraphrasing the article, if we have two consecutive fractions $a/b$ and $c/d$, then the next fraction is $p/q$ where

$$
\begin{aligned}
	p &= \lfloor \frac{n + b}{d} \rfloor c - a
	\\
	q &= \lfloor \frac{n + b}{d} \rfloor d - b
\end{aligned}
$$
We already have one of the fractions we start with, namely $1/3$, but we need the fraction immediately to its left. But we've done that already in [#71 - Ordered fractions](/blog/project_euler/2017-06-21-071---Ordered-fractions){:.heading.flip-title}. All we need to do is slightly adapt, then continue generating the next fraction until we hit $1/2$.

I've created a general function following \#71's logic to find the fraction immediately to the left of $a/b$, given the maximum denominator. 
```python
# file: "problem073.py"
def findClosestFraction(n, a, b):
    minDist = float('inf')
    bestNum = 0
    bestDenom = 1
    for denom in range(3, n):
        # Skip multiples of b
        if denom % b == 0:
            continue
        # Calculate closest
        num = a * denom // b
        dist = a/b - num/denom
        if dist < minDist:
            minDist = dist
            bestNum = num
            bestDenom = denom
    return bestNum, bestDenom

n = 12000
a, b = findClosestFraction(n, 1, 3)

count = 0
c, d = 1, 3
while c/d != 1/2:
    k = (n + b) // d
    a, b, c, d = c, d, k * c - a, k * d - b
    count += 1

print(count - 1)
```
Running this code gives us,
```
7295372
2.8874196000397205 seconds.
```
Therefore, when $d=12000$, we have **7295372** fractions between $1/3$ and $1/2$.

