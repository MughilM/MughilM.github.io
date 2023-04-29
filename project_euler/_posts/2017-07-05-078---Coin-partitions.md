---
layout: post
title: "#78 - Coin partitions"
date: 2017-07-05 14:32
number: 78
tags: [30_diff]
---
> Let $p(n)$ represent the number of different ways in which $n$ coins can be separated into piles. for example, five coins can be separated into piles in exactly seven different ways, so $p(5)=7$.
> 
> $$
> \bigcirc\bigcirc\bigcirc\bigcirc\bigcirc
> \\
> \bigcirc\bigcirc\bigcirc\bigcirc\quad\bigcirc
> \\
> \bigcirc\bigcirc\bigcirc\quad\bigcirc\bigcirc
> \\
> \bigcirc\bigcirc\bigcirc\quad\bigcirc\quad\bigcirc
> \\
> \bigcirc\bigcirc\quad\bigcirc\bigcirc\quad\bigcirc
> \\
> \bigcirc\bigcirc\quad\bigcirc\quad\bigcirc\quad\bigcirc
> \\
> \bigcirc\quad\bigcirc\quad\bigcirc\quad\bigcirc\quad\bigcirc
> $$
> 
> Find the least value of $n$ for which $p(n)$ is divisible by one million.
{:.lead}
* * *

This is actually the same problem as [#76 - Counting summations](/blog/project_euler/2017-06-21-076-Counting-summations){:.heading.flip-title}, just worded differently. This time, we are counting all the coins as "one pile". Regardless, we can use the same method for generating $p(n)$. We now have to append to a partition list, and our stopping condition is different. 

Since we are checking when $p(n)$ is divisible by one million, we check when $p(n)\equiv 0\mod 1000000$. Modulus distributes through sums, so we do not keep the potentially large values of $p(n)$ as we go along, and instead store $p(n)\mod 1000000$.
```python
# file: "problem078.py"
partitions = [1]
pent = lambda x: x * (3 * x - 1) // 2
n = 1
while partitions[-1] != 0:
    k = 1
    currP = 0
    pentk = pent(k)
    while pentk <= n:
        currP = (currP + partitions[n - pentk] * (-1) ** (k + 1)) % 1000000
        # If k is positive, then it turns into
        # its negative counterpart,
        # Otherwise, it goes to the next number
        if k > 0:
            k *= -1
        else:
            k = k * -1 + 1
        pentk = pent(k)
    # Append...
    partitions.append(currP)
    n += 1

print('n =', len(partitions) - 1, 'is when p(n) is divisible by 1000000.')
```
The output is,
```
n = 55374 is when p(n) is divisible by 1000000.
16.1984474 seconds.
```
Thus, **55374** coins is the fewest number needed. As opposed to a list, we could have also stored a set that maps the coins to the number of piles, for quick look up.