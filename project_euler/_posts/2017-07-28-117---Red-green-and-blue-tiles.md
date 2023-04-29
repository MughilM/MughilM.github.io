---
layout: post
title: "#117 - Red, green and blue tiles"
date: 2017-07-28 10:36
number: 117
tags: [35_diff]
---
> Using a combination of grey square tiles and oblong tiles chosen from: red tiles (measuring two units), green tiles (measuring three tiles), and blue tiles (measuring four tiles), it is possible to tile a row measuring five units in length in exactly fifteen different ways.
> 
> ![p117](/assets/img/project_euler/p117.png){:style="display:block; margin-left:auto; margin-right:auto"}
> 
> How many ways can a row measuring fifty units in length be tiled?
> 
> This is related to [#116 - Red, green or blue tiles](/blog/porject_euler/2017-07-28-116-Red-green-or-blue-tiles){:.heading.flip-title}
> {:.note}
{:.lead}
* * *

Since the red block is the smallest we have at 2 units, and the largest is the blue tile at 4 units, it is essentially like [#114 - Counting block combinations I](/blog/project_euler/2017-07-27-114-Counting-block-combinations-I){:.heading.flip-title} where the minimum tile length is 2 and the maximum tile length is 4. One pickle is that there should be no gaps between the tiles. However, this is accounted for by controlling the remaining black tiles for placement. The same concept of dynamic programming applies. 

For the code, notice the `blackSize = np.arange(n - blockSize + 1)` line. Previously, we had it start at -1, because of how the problem of "one gap required" was set up. Now, our array actually starts at 0.
```python
# file: "problem117.py"
targetN = 50
nVals = np.zeros(targetN+1, dtype=np.int64)
# 0 is 1
nVals[0] = 1
# Start from n = 0, and find values up to target
for n in range(0, targetN+1):
    configs = 0
    # From the smallest red block (2) to the largest
    # (4), move it across and find the number of
    # configurations you can put in the remaining
    # usable black blocks
    for blockSize in range(2, 5):
        blackSizes = np.arange(n - blockSize + 1)
        # Sum the values at those locations
        # Add 1
        configs += np.sum(nVals[blackSizes])
    # Add 1 to final count because of all black
    nVals[n] = configs + 1

print(nVals[-1])
```
Running this code, we get 
```
100808458960497
0.0025583000000000133 seconds.
```
Therefore, we have **100808458960497** ways of tiling a row 50 units in length.