---
layout: post
title: "#114 - Counting block combinations I"
date: 2017-07-27 16:35
number: 114
tags: [35_diff]
---
> A row measuring seven units in length has red blocks with a minimum length of three units placed on it, such that any two red blocks (which are allowed to be different lengths) are separated by at least one grey square. There are exactly seventeen ways of doing this.
> 
> ![p114](/assets/img/project_euler/p114.png){:style="display:block; margin-left:auto; margin-right:auto"}
> 
> How many ways can a row measuring fifty units in length be filled?
> 
> Although the example does not lend itself to the possibility, in general it is permitted to mix block sizes. For example, on a row measuring eight units in length you could use red (3), grey (1), and red (4).
> {:.note}
{:.lead}
* * *

This type of problem screams dynamic programming. Why DP? If you were to go about this normally, you first place one red block somewhere on the row. Of the black units remaining, **how many ways can you put red blocks on the remaining black tiles?** This is where it would be handy if you had the previous value to look up.

How about configurations that are symmetric (such as the red (3), gray (1), and red (3) above)? To avoid double counting, we will only worry about the number of ways of placing blocks in the black tiles **to the left of the placed red block.**

To summarize,
* Choose a red block length $\ell_r$. Let the length of the row be $n$.
* Slide the red block across the row until the end. This is exactly $n-\ell_r + 1$ steps.
* For each position $i$ (0-indexed), take the number of **valid** black tiles to the left, and look up how many ways are there for that amount of tiles to be filled. Under this problem, there needs to be a gap of one, so the number of valid tiles is $i-1$. We also need to add 1 to count the configuration of all black tiles

Our dynamic programming equation is,

$$
F(n) = \begin{cases}
	1 & n < 3
	\\
	\displaystyle\sum_{r=3}^n\sum_{i=0}^{n-r}F(n-1) + 1 & n \geq 3
\end{cases}
$$
Again, I'm using `numpy`'s advanced indexing features.
```python
# file: "problem114.py"
targetN = 50
nVals = np.zeros(targetN+2, dtype=np.int64)
# -1,0,1,2 are all 1
nVals[0] = 1
# Start from n = 3, and find values up to target
for n in range(0, targetN+1):
    configs = 0
    # From the smallest red block (3) to the largest
    # (n), move it across and find the number of
    # configurations you can put in the remaining
    # usable black blocks
    for redSize in range(3, n+1):
        blackSizes = np.arange(-1, n - redSize)
        # Sum the values at those locations
        # Add 1
        configs += np.sum(nVals[blackSizes + 1])
    # Add 1 to final count because of all black
    nVals[n + 1] = configs + 1

print(nVals[-1])
```
Running this short loop results in,
```
16475640049
0.022253200000000084 seconds.
```
Thus, there are **16475640049** ways to fill a row measuring 50 units long.