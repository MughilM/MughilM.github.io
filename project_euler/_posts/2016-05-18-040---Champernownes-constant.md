---
layout: post
title: "#40 - Champernowne's constant"
date: 2016-05-18 19:49
number: 40
tags: [05_diff]
---
> An irrational decimal fraction is created by concatenating the positive integers:
> 
> <p align="center">
> 0.12345678910<span style="color:red">1</span>112131415161718192021...
> </p>
> 
> It can be seen that the 12th digit of the fractional part is 1.
> 
> If $d_n$ represents the $n^{\text{th}}$ digit of the fractional part, find the value of the following expression:
>
> $$
> d_1\times d_{10}\times d_{100}\times d_{1000}\times d_{10000}\times d_{100000}\times d_{1000000}
> $$
{:.lead}
* * *

Nothing overly fancy we need to do. Just loop through all the numbers, keeping track of which digit we are at.
```python
# file: "problem040.py"
num = 1
prod = 1
d = 1
while (d < 1000001):
    for i in range(len(str(num))):
        if (d == 1 or d == 10 or
            d == 100 or d == 1000 or
            d == 10000 or d == 100000 or
            d == 1000000):
            prod = prod * int(str(num)[i:i+1])
        d = d + 1
    num = num + 1
print(prod)
```
The output is,
```
210
0.5373719607973275 seconds.
```
Thus, the product is **210**.