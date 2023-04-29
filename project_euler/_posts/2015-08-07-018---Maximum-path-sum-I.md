---
layout: post
title: "#18 - Maximum path sum I"
date: 2015-08-07 20:47
number: 18
tags: [05_diff, Dynamic programming]
---
> By starting at the top of the triangle below and moving to adjacent numbers on the row below, the maximum total from top to bottom is 23.
> 
> <pre style="text-align:center">
> <span style="color:red"><b>3</b></span>
> <span style="color:red"><b>7</b></span> 4
> 2 <span style="color:red"><b>4</b></span> 6
> 8 5 <span style="color:red"><b>9</b></span> 3
> </pre>
> 
> That is, 3 + 7 + 4 + 9 = 23.
> 
> Find the maximum total from top to bottom of the triangle below:
> 
> <pre style="text-align:center">
> 75
> 95 64
> 17 47 82
> 18 35 87 10
> 20 04 82 47 65
> 19 01 23 75 03 34
> 88 02 77 73 07 63 67
> 99 65 04 28 06 16 70 92
> 41 41 26 56 83 40 80 70 33
> 41 48 72 33 47 32 37 16 94 29
> 53 71 44 65 25 43 91 52 97 51 14
> 70 11 33 28 77 73 17 78 39 68 17 57
> 91 71 52 38 17 14 91 43 58 50 27 29 48
> 63 66 04 68 89 53 67 30 73 16 69 87 40 31
> 04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
> </pre>
> 
> As there are only 16384 routes, it is possible to solve this problem by trying every route. However, Problem 67, is the same challenge with a triangle containing one-hundred rows; it cannot be solved by brute force, and requires a clever method! ;o)
> {:.note}
{:.lead}
* * *

Let's code the "clever method" from the start, so that Problem 67 will be easy. Notice that we don't need the path itself, just the sum. Just like with the [#15 - Lattice paths](/blog/project_euler/2015-08-07-015-Lattice-paths){:.heading.flip-title}, we can compute the largest sum in each row as we go. At a given row, we compare the max sum to the left, and to the right above, and sum whichever sum would be greater to the current row. Here's how we do it with the given example in the 2nd and 3rd iteration.

<pre style="text-align:center">
10 7
2 4 200
8 5 9 3
</pre>

<pre style="text-align:center">
12 14 207
8 5 9 3
</pre>

The final row is,
<pre style="text-align:center">
20 19 216 210
</pre>

We see the greatest number in the last row is 216. We can actually go in the other direction, comparing sums from below. In this manner, we end with one number at the top, so I'll be using this method.

<pre style="text-align:center">
3
7 4
10 13 209
</pre>

<pre style="text-align:center">
3
20 213
</pre>

<pre style="text-align:center">
216
</pre>

I have saved the triangle in `problem018.txt`. Te make reading the file easier, I have placed numbers which are below and to the left in the triangle directly below it in the text file, like so:

```
75
95 64
17 47 82
18 35 87 10
...
```

```python
# file: "problem018.py"
# open file and convert to integers and make 2d array.
with open("problem018.txt") as f:
    triangle = [list(map(int, line.split(' '))) for line in f.read().splitlines()]
# Work bottom to top, starting from the second to last
# row so we can numbers below.
for i in range(len(triangle) - 2, -1, -1):
    for j in range(len(triangle[i])):
        # The number to the "bottom left" is really just below,
        # in terms of indices, and "bottom right" is diagonally right.
        triangle[i][j] = triangle[i][j] + max(triangle[i + 1][j], triangle[i + 1][j + 1])
# Grab the top number.
print(triangle[0][0])
```
Running our loop gives,
```
1074
0.0003338271604938272 seconds.
```
Therefore, our needed sum is **1074**.
