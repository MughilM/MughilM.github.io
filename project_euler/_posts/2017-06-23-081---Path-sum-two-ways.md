---
layout: post
title: "#81 - Path sum: two ways"
date: 2017-06-23 10:24
number: 81
tags: [10_diff]
---
> In the 5 by 5 matrix below, the minimal path sum from the top left to the bottom right, by **only moving to the right and down**, is indicated in bold red and is equal to 2427.
> 
> $$
> \begin{pmatrix}
> 	\color{red}{\mathbf{131}} & 673 & 234 & 103 & 18
> 	\\
> 	\color{red}{\mathbf{201}} & \color{red}{\mathbf{96}} & 
> 		\color{red}{\mathbf{342}} & 965 & 150
> 	\\
> 	630 & 803 & \color{red}{\mathbf{746}} & \color{red}{\mathbf{422}} & 111
> 	\\
> 	537 & 699 & 497 & \color{red}{\mathbf{121}} & 956
> 	\\
> 	805 & 732 & 524 & \color{red}{\mathbf{37}} & \color{red}{\mathbf{331}}
> \end{pmatrix}
> $$
> 
> Find the minimal path sum from the top left to the bottom right by only moving right and down in [matrix.txt](https://projecteuler.net/project/resources/p081_matrix.txt) (right click and "Save Link/Target As..."), a 31K text file containing an 80 by 80 matrix.
{:.lead}
* * *

The addition in this matrix is a one-way street, since we're only dealing with positive integers. The sum will only get bigger. We can then save the **minimum path sum** for *any* path ending in a cell by comparing adding the number to its right and below it. I'll show an example using a smaller matrix. Suppose our matrix $M$ is just the top 3 by 3 corner of the example:

$$
M = \begin{pmatrix}
	131 & 673 & 234
	\\
	201 & 96 & 342
	\\
	630 & 803 & 746
\end{pmatrix}
$$
We also have another matrix $P$ where each element $p_{ij}$ is the minimum path sum of all paths in $M$ ending at the cell in row $i$ and column $j$. Since we only move right or down, the only paths that end in the top row are the ones that only move right.

$$
\begin{aligned}
	p_{12} &= m_{11} + m_{12} = 131 + 673 = 804
	\\
	p_{13} &= m_{11} + m_{12} + m_{13} = 131 + 673 + 234 = 1038
\end{aligned}
$$
We can make the same deduction of the first column. Our $P$ matrix is now

$$
P = \begin{pmatrix}
	131 & 804 & 1038 \\
	332 & * & * \\
	962 & * & *
\end{pmatrix}
$$

For $p_{22}$, **all paths ending in the middle cell had to have come from the cell above or the cell to the left**. But wait, we already calculated the minimum path sums for those cells! In that case, we need to add those cells and save whichever is smaller. In general,

$$
p_{ij} = \min\begin{cases}p_{i-1,j}+m_{ij} \\ p_{i,j-1}+m_{ij}\end{cases}
$$
We can quickly fill in $P$:

$$
P = \begin{pmatrix}
	131 & 804 & 1038 \\
	332 & 428 & 770 \\
	962 & 1231 & 1516
\end{pmatrix}
$$
Notice that we are not saving the actual path as we go along, since the problem didn't ask for that. Additionally, during coding, we can overwrite the additional matrix itself, since we only visit each cell once. 
```python
# file: "problem081.py"
with open('p081_matrix.txt') as f:
    matrix = [list(map(int, line.split(','))) for line in f.read().splitlines()]

n = 80
# First fill in cumulative sums on first row
# and first column
for i in range(1, n):
    matrix[0][i] += matrix[0][i - 1]
for i in range(1, n):
    matrix[i][0] += matrix[i - 1][0]
# Now update all the cells with the minimum sum
for i in range(1, len(matrix)):
    for j in range(1, len(matrix[i])):
        matrix[i][j] += min(matrix[i - 1][j], matrix[i][j - 1])

print(matrix[-1][-1])
```
Running this results in an output of,
```
427337
0.005567600000000006 seconds.
```
Thus, the minimum sum in our 80 by 80 matrix is **427337**.