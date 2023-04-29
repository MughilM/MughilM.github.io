---
layout: post
title: "#53 - Combinatoric selections"
date: 2016-06-17 14:11
number: 53
tags: [05_diff]
---
> There are exactly ten ways of selecting three from five, 12345:
> 
> $$
> 123,\,\,124,\,\,125,\,\,134,\,\,135,\,\,145,\,\,234,\,\,235,\,\,245,\,\,\text{and }345
> $$
> 
> In combinatorics, we use the notation $_5 C_3=10$.
> 
In general $_nC_r=\frac{n!}{r!(n-r)!}$, where $r\leq n, n!=n\times(n-1)\times\dots\times 3\times 2\times 1$, and $0!=1$.
> 
> How many, not necessarily distinct, values of $_nC_r$ for $1\leq n\leq 100$, are greater than one-million?
{:.lead}
* * *

Notice that the formula for $_nC_r$ has three factorials. Recomputing each of these factorials each time will be time-consuming and inefficient. Instead, we can find $n!$ for all $0\leq n\leq 100$, and grab the values we need for each computation. Python's `math` package has a quick factorial function we can use:

```python
# file: "problem053.py"
factorials = [math.factorial(i) for i in range(0, 101)]
count = 0
for n in range(23, 101):
    for r in range(n + 1):
        if factorials[n] / (factorials[r] * factorials[n - r]) > 1000000:
            count += 1
print(count)
```

Running this short loop results in,

```
4075
0.0038949089640585117 seconds.
```
Thus, there are **4075** values that are greater than one million.
## Bonus
We can actually solve this problem without computing a single factorial! We can use **Pascal's triangle** to find $_nC_r$, as the rows of the triangle are the values of $_nC_r$. Pascal's triangle is constructed in the following manner:

1. The first row is $1$. 
2. The second row is $1\quad 1$.
3. Each row after the second is formed by placing two ones at the ends, while the middle values are calculated by adding the two numbers above. For example, the first 5 rows are shown below.

<pre style="text-align:center">
1
1 1
1 2 1
1 3 3 1
1 4 6 4 1
</pre>

If the rows are numbered starting from 0, and the elements in the row are also numbered starting from 0, the $r^\text{th}$ number in the $n^\text{th}$ row is exactly $_nC_r$. Using the generative nature of Pascal's triangle, we can continuously generate each row until the 100th row, and count the numbers which are greater than one-million.

```python
currRow = [1, 1]
count = 0
for n in range(2, 101):
    nextRow = [0] * (n + 1)
    nextRow[0] = 1
    nextRow[-1] = 1
    # Make the next row
    for i in range(len(currRow) - 1):
        nextRow[i+1] = currRow[i] + currRow[i+1]
    # Count how many are bigger than 1 million
    count += len([x for x in nextRow if x > 1000000])
    currRow = nextRow
print(count)
```

We have to start from the beginning in order to see what the 23rd row is. Regardless, the output of is the same, although slightly faster.

```
4075
0.0015237512804923096 seconds.
```