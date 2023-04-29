---
layout: post
title: "#75 - Singular integer right triangles"
date: 2018-06-03 16:16
number: 75
tags: [25_diff]
---
> It turns out that 12 cm is the smallest length of wire that can be bent to form an integer sided right angle triangle in exactly one way, but there are many more examples.
> 
> $$
> \mathbf{12}\textbf{ cm}: (3,4,5)
> \\
> \mathbf{24}\textbf{ cm}: (6,8,10)
> \\
> \mathbf{30}\textbf{ cm}: (5,12,13)
> \\
> \mathbf{36}\textbf{ cm}: (9,12,15)
> \\
> \mathbf{40}\textbf{ cm}: (8,15,17)
> \\
> \mathbf{48}\textbf{ cm}: (12,16,20)
> $$
> 
> In contrast, some lengths of wire, like 20 cm, cannot be bent to form an integer sided right angle triangle, and other lengths allow more than one solution to be found; for example, using 120 cm it is possible to form exactly three different integer sided right angle triangles.
> 
> $$
> \mathbf{120}\textbf{ cm}: (30,40,50), (20,48,52), (24,45,51)
> $$
> 
> Given that $L$ is the length of the wire, for how many values of $L\leq 1\,500\,000$ can exactly one integer sided right angle triangle be formed?
{:.lead}
* * *

[#39 - Integer right triangles](/blog/project_euler/2016-05-17-039-Integer-right-triangles){:.heading.flip-title} is extremely similar to this. Like that problem, we will generate the sides of the triangles first. The method of generating the triples will be the same as that problem. Please consult that write-up on the actual method of generation.

The method only generates primitive triples however. We need to take multiples in order to properly check anything. We calculate each perimeter and add to a list. Then `np.unique()` along with `return_counts=True` will allow us to which values occur only once. 
```python
# file: "problem075.py"
maxi = 1500000
sums = []
for triple in genTriples(np.array([3, 4, 5]), maxi=maxi):
    primPerim = np.sum(triple)
    sums.extend(range(primPerim, maxi + 1, primPerim))
uniques, counts = np.unique(sums, return_counts=True)
exactlyOnce = uniques[np.where(counts == 1)]
print(exactlyOnce.size)
```
Running the above results in an output of,
```
161667
2.8955301999999996 seconds.
```
Thus, there are **161667** values under 1.5 million such that there exists exactly one integer-sided right angle triangle.