---
layout: post
title: "#15 - Lattice paths"
date: 2015-08-07 20:44
number: 15
tags: [05_diff, Dynamic programming]
---
> Starting in the top left corner of a 2 x 2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.
> 
> ![Image not found: /assets/img/project_euler/p015.png](/assets/img/project_euler/p015.png){:style="display:block; margin-left:auto; margin-right:auto"}
> 
> Lattice paths
> {:.figcaption}
> 
> How many such routes are there through a 20 x 20 grid?
{:.lead}
* * *

We can use dynamic programming here, since we have optimal substructure. The rule of the number of paths ending at a single cell does not change depending on where you are at. Meaning, **if we had the number of paths for the cell immediately to the left, and above, all we have to do is add them.**. In this way, we have turned this into a very fast solution, at the expense of storing the entire grid.
```python
# file: "problem015.py"
size = 20
# It's (size + 1) because a size x size grid
# has (size + 1) intersections across the top and
# down.
grid = np.zeros((size + 1, size + 1), dtype=object)
# The top row and left column only
# have 1 way to get there.
grid[0] = 1
grid[:, 0] = 1
# For each inner grid point, add
# the number to its left and up
for i in range(1, size + 1):
    for j in range(1, size + 1):
        grid[i, j] = grid[i - 1, j] + grid[i, j - 1]
# The number in the bottom right corner is what
# we want.
print(grid[-1, -1])
```
Running our quick loop,
```
137846528820
0.0002860246913580247 seconds.
```
Therefore, our answer is **137846528820**.