---
layout: post
title: "#39 - Integer right triangles"
date: 2016-05-17 10:50
number: 39
tags: [05_diff]
---
> If $p$ is the perimeter of a right angle triangle with integral length sides, $\{a,b,c\}$, there are exactly three solutions for $p = 120$.
> 
> $$
> \begin{aligned}
> \{20,48,52\},\{24,45,51\},\{30,40,50\}
> \end{aligned}
> $$
> 
> For which value of $p\leq1000$, is the number of solutions maximised?
{:.lead}
* * *

There are two possible we could go about this: Either we test each $p$ and see which right triangles exist, or we take all right angles with perimeters below 1000 and see which perimeter occurs the most. Personally, I think the latter is the easier way to go, as not every $p$ will have an integer right triangle associated with it.

So then how do we find all Pythagorean triples with perimeters less than or equal 1000? After some research on methods of generating triples, I came across [this Wikipedia page](https://en.wikipedia.org/wiki/Tree_of_primitive_Pythagorean_triples), which details 3 matrices which can be used to find all triples. I have reproduced the matrices below.

$$
\begin{aligned}
A &= \begin{bmatrix}
	1 & -2 & 2 \\
	2 & -1 & 2 \\
	2 & -2 & 3
\end{bmatrix} \\
B &= \begin{bmatrix}
	1 & 2 & 2 \\
	2 & 1 & 2 \\
	2 & 2 & 3
\end{bmatrix} \\
C &= \begin{bmatrix}
	-1 & 2 & 2 \\
	-2 & 1 & 2 \\
	-2 & 2 & 3
\end{bmatrix}
\end{aligned}
$$

To generate triples, start with the most basic of them: (3,4,5). Convert this to a vector $\mathbf{v}=\langle 3,4,5 \rangle$. Now multiply this vector by one of the three matrices above on the left i.e. $A\mathbf{v}$. The result of this is another Pythagorean triple! What's more, each of the three matrices will always get you 3 distinct triples. For example, $A\mathbf{v}$ will get you the triple $\langle 5,12,13 \rangle$, while $B\mathbf{v}=\langle 21,20,29\rangle$ and $C\mathbf{v}=\langle 15,8,17\rangle$.

The order also matters during multiplication, as evidenced by the tree structure. For example, multiplying $C$, then $B$, then $A$ gets you $ABC\mathbf{v}=\langle 115,252,277\rangle$.

This method will only generate **primitive** Pythagorean triples i.e. $gcd(a,b,c) = 1$ for each triple e.g. $\langle 10, 24, 26\rangle$, although a triple, will not be generated.
{:.note}

We can adapt recursion to this tree structure easily. We recurse down for each matrix, and as soon as we encounter a triple that is larger than our perimeter, we cut the recursion. I'll also wrap the recursion in a generator, so that we won't have to store all the triples at once. 
```python
# file: "problem039.py"
def genPythagoreanTriples(triple, maxP, A, B, C):
    # Base case...
    if sum(triple) > maxP:
        return
    yield triple
    # Multiply each matrix
    for matrix in [A, B, C]:
        for multTriple in genPythagoreanTriples(np.dot(matrix, triple), maxP, A, B, C):
            yield multTriple
```
In order to test for non-primitive triples, we can keep multiplying our triple until we exceed the perimiter. For $\langle 3, 4, 5\rangle$, the maximum multiple will be $\langle 249, 332, 415\rangle$ with a perimeter of 996. The `np.unique` function can be used to gather up unique perimeter values, as well as the number of times we occur.
```python
# file: "problem039.py"
# Make the matrices
A = np.array([
    [1, -2, 2],
    [2, -1, 2],
    [2, -2, 3]
])
B = np.array([
    [1, 2, 2],
    [2, 1, 2],
    [2, 2, 3]
])
C = np.array([
    [-1, 2, 2],
    [-2, 1, 2],
    [-2, 2, 3]
])

maxP = 1000
trianglePerims = []
for triple in genPythagoreanTriples(np.array([3,4,5]), maxP, A, B, C):
    perimTriple = sum(triple)
    # Generate all triple multiples that don't exceed maxP
    # and add it to the list
    for i in range(1, maxP // perimTriple + 1):
        trianglePerims.append(i * perimTriple)

# Find the unique solutions
unique, count = np.unique(trianglePerims, return_counts=True)
# Find the p with the most solutions
print('p =', unique[np.argmax(count)], 'has', np.max(count), 'solutions.')
```
Running the code results in an output of,
```
p = 840 has 8 solutions.
0.0012768364795974804 seconds.
```
Therefore, $\mathbf{p=840}$ is the perimeter with the most solutions (8).