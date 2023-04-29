---
layout: post
title: "#85 - Counting rectangles"
date: 2017-06-25 08:54
number: 85
tags: [15_diff]
---
> By counting carefully it can be seen that a rectangular grid measuring 3 by 2 contains eighteen rectangles:
> 
> ![rectImage](/assets/img/project_euler/p085){:style="display:block; margin-left:auto; margin-right:auto"}
> 
> Although there exists no rectangular grid that contains exactly two million rectangles, find the area of the grid with the nearest solution.
{:.lead}
* * *

The image shows an approach on how we count the rectangles. We are taking each possible size of the "sub-rectangles", and counting how many there are. In this case, a 2 by 1 rectangle is **different** than a 1 by 2 rectangle.

Given a grid size $m\times n$, for each possible "sub-rectangle" of dimension $i\times j$, we'll count how many these rectangles are contained in the larger grid. We can fit $m-i+1$ rectangles lengthwise, and $n-j+1$ rectangles widthwise before we run out of room on each side. In total, that means there are $(m-i+1)(n-j+1)$ sub-rectangles of dimension $i\times j$ that can be fit in an $m\times n$ grid. Now we have to sum over all possible values $i$ and $j$. They both grow up till the original grid size. If $R(m,n)$ is the number of rectangles, then

$$
R(m,n) = \sum_{i=1}^m\sum_{j=1}^n (m-i+1)(n-j+1)
$$
Let's simplify this expression a bit more. Notice that each product only depends on one of the variables, and not both at the same time. That means we can convert this sum of products into a product of sums. AFter a little alegbra, we have,

$$
\begin{aligned}
	R(m,n) &= \sum_{i=1}^m\sum_{j=1}^n(m-i+1)(n-j+1)
	\\ &=
	\left(\sum_{i=1}^m (m-i+1)\right)
		\left(\sum_{j=1}^n (n-j+1)\right)
	\\ &=
	\left(\sum_{i=1}^m(m+1) - \sum_{i=1}^m i\right)
		\left(\sum_{j=1}^n(n+1) - \sum_{j=1}^n j\right)
	\\ &=
	\left(m(m+1) - \frac{m(m+1)}{2}\right)
		\left(n(n+1) - \frac{n(n+1)}{2}\right)
	\\ &=
	\left(\frac{m(m+1)}{2}\right)\left(\frac{n(n+1)}{2}\right)
	\\ &=
	\boxed{\frac{mn(m+1)(n+1)}{4}}
\end{aligned}
$$
We now have a direct formula for the number of rectangles in a grid. We can either keep increasing the maximum size and break until we find our solution, or we can set an upper limit and hope it's
smaller. I went with the latter, and had $m$ go until 100.
```python
# file: "problem085.py"
minDist = float('inf')
target = 2000000
area = 0
bestM = 0
bestN = 0
for m in range(1, 101):
    for n in range(1, m+1):
        rects = m * n * (m + 1) * (n + 1) // 4
        if math.fabs(rects - target) < minDist:
            minDist = math.fabs(rects - target)
            area = m * n
            bestM = m
            bestN = n
rects = bestM * bestN * (bestM + 1) * (bestN + 1) // 4
print('The grid is {} x {} (Area = {}) with {} rectangles.'.format(bestM, bestN, area, rects))
```
Running this code, we get an output of,
```
The grid is 77 x 36 (Area = 2772) with 1999998 rectangles.
0.0024605999999999795 seconds.
```
Thus, a $77\times 36$ grid has the closest to 2 million rectangles, and its area is **2772**.