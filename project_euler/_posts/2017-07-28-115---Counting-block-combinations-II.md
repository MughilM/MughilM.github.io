---
layout: post
title: "#115 - Counting block combinations II"
date: 2017-07-28
number: 115
tags: [35_diff]
---
> This is a much more difficult version of [#114 - Counting block combinations I](/blog/project_euler/2017-07-27-114-Counting-block-combinations-I){:.heading.flip-title}
> {:.note}
> 
> A row measuring $n$ units in length has red blocks with a minimum length of $m$ units placed on it, such that any two red blocks (which are allowed to be different lengths) are separated by at least one black square.
>
> Let the fill-count function, $F(m,n)$, represent the number of ways that a row can be filled.
>
> For example, $F(3,29) = 673135$ and $F(3, 30) = 1089155$.
>
> That is, for $m=3$, it can be seen that $n=30$ is the smallest value for which the fill-count function first exceeds one-million.
>
> In the same way, for $m=10$, it can be verified that $F(10, 56) = 880711$ and $F(10, 57) = 1148904$, so $n=57$ is the least value for which the fill-count function first exceeds one million.
>
> For $m=50$, find the least value of $n$ for which the fill-count function first exceeds one million.
{:.lead}
* * *

The only difference between this problem and [#114 - Counting block combinations I](/blog/project_euler/2017-07-27-114-Counting-block-combinations-I){:.heading.flip-title} is the addition of the minimum block size. In 114, we used a minimum red block size of 3, but here they are asking for 50. In terms of code, we only change the line where we test each possible red block size. Additionally, we now need to keep calculating until we have an amount of one million.
```python
# file: "problem115.py"
nVals = np.array([], dtype=np.uint64)
# -1,0,1,2 are all 1
nVals = np.append(nVals, 0)
# Start from n = 0, and find values up to target
n = 0
minRedBlockSize = 50
while nVals[-1] < 1000000:
    configs = 0
    # From the smallest red block (50) to the largest
    # (n), move it across and find the number of
    # configurations you can put in the remaining
    # usable black blocks
    for redSize in range(minRedBlockSize, n+1):
        blackSizes = np.arange(-1, n - redSize)
        # Sum the values at those locations
        # Add 1
        configs += np.sum(nVals[blackSizes + 1])
    # Add 1 to final count because of all black
    nVals = np.append(nVals, configs + 1)
    n += 1

print(n-2)
print(int(nVals[-1]))
```
Running, we get
```
168
1053389
0.0759726 seconds.
```
Thus, we need a minimum row length of **168** units before we exceed one million ways to fill it. 