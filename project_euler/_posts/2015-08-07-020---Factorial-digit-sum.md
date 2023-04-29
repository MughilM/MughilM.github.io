---
layout: post
title: "#20 - Factorial digit sum"
date: 2015-08-07 20:51
number: 20
tags: [05_diff]
---
> $n!$ means $n\times (n-1)\times\cdots\times 3\times 2\times 1$.
> 
> For example, $10! = 10\times 9\times\cdots\times 3\times 2\times 1 = 3628800$, and the sum of the digits in the number $10!$ is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.
> 
> Find the sum of the digits in the number $100!$.
{:.lead}
* * *

As said before, Python can very easily handle large numbers. We write a simple loop to calculate $n!$ (though there is one built into the `math` package). To extract the digits, we convert the number into a string, and `map` the `int` function in each element.
```python
# file: "problem020.py"
def factorial(n):
    product = 1
    for i in range(2, n + 1):
        product *= i
    return product
n = 100
fact = factorial(n)
s = sum([int(chara) for chara in str(fact)])
print(s)
```
At the end, the output is
```
648
9.362962962962964e-05 seconds.
```
Thus, the sum is **648**.