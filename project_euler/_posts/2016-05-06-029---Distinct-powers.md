---
layout: post
title: "#29 - Distinct powers"
date: 2016-05-06 22:44
number: 29
tags: [05_diff]
---
> Consider all integer combinations of $a^b$ for $2\leq a\leq 5$ and $2\leq b\leq 5$:

> | $a^b$ | 2  | 3   | 4   | 5    |
|-------|----|-----|-----|------|
| **2** | 4  | 8   | 16  | 32   |
| **3** | 9  | 27  | 81  | 243  |
| **4** | 16 | 64  | 256 | 1024 |
| **5** | 25 | 125 | 625 | 3125 |

> If they are then placed in numerical order, with any repeats removed, we get the following sequence of 15 distinct terms:
> 
> <p align="center">
>     4, 8, 9, 16, 25, 27, 32, 64, 81, 125, 243, 256, 625, 1024, 3125
> </p>
> 
> How many distinct terms are in the sequence generated by $a^b$ for $2\leq a\leq 100$ and $2\leq b\leq 100$?
{:.lead}
* * *

We can solve this through brute force directly. The `set()` structure automatically removes duplicates for us. A simple for loop suffices:
```python
# file: "problem029.py"
s = set()

for a in range(2, 101):
    for b in range(2, 101):
        s.add(a ** b)

print(len(s))
```
Running the short loop produces an output of,
```
9183
0.02056965463349574 seconds.
```
Therefore, there are **9183** distinct terms in the sequence.