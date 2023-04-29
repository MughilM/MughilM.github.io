---
layout: post
title: "#34 - Digit factorials"
date: 2016-05-09 17:27
number: 34
tags: [05_diff]
---
> 145 is a curious number, as 1! + 4! + 5! = 1 + 24 + 120 = 145.
> 
> Find the sum of all numbers which are equal to the sum of the factorial of their digits.
> 
> As 1! = 1 and 2! = 2 are not sums they are not included.
> {:.note}
{:.lead}
* * *

We need an upper bound for this problem. Suppose our test number $n$ has $k$ digits. The largest $n$ can be is a set of $k$ 9's, in which the sum of the digits will be $k\times 9!$. If $k=8$, then the sum is $8\times 9! = 2\,903\,040$, which is only 7 digits. Therefore, $n$ can't be more than 7 digits. Additionally, since $7\times 9! = 2\,540\,160$, we check until this number.
```python
# file: "problem034.py"
def factSum(n):
    s = 0
    for d in str(n):
        s += math.factorial(int(d))
    return s

s = 0

for i in range(10, 2540161):
    if factSum(i) == i:
        s = s + i
print(s)
```
The output is,
```
40730
8.536330512194187 seconds.
```
Therefore, our sum is **40730**. We also see that we could have safely ignored all 6 and 7 digit numbers as well.