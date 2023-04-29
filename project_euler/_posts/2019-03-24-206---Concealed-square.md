---
layout: post
title: "#206 - Concealed Square"
date: 2019-03-24 11:17
number: 206
tags: [05_diff]
---
> Find the unique positive integer whose square has the form 1\_2\_3\_4\_5\_6\_7\_8\_9\_0, where each "\_" is a single digit.
{:.lead}
* * *

Short and sweet. The first thing to note is that the square ends in a 0. The only square numbers that end in 0 are multiples of 100, which means the final blank is a 0. We can now instead look for a square that looks like 1\_2\_3\_4\_5\_6\_7\_8\_9.

We want a square that ends in 9, which means the square root of this number ends in a 3 or 7 (since $3^2=9$ and $7^2=49$, both end in 9s). We can brute force this problem.

The largest possible number is where only 9s are filled in i.e. 19293949596979899.The largest square less than this is $138\,902\,662^2=19\,293\,949\,510\,686\,244$. We start with 138902657, since we want a number ending in 3 or 7. Then we alternatively subtract 4 and 6 to keep the last digit 3 or 7. A short function to test if the square follows the pattern is also needed.
```python
# file: "problem206.py"
def isValid(n):
    # Only need to check until the "8"
    # because our number ends in "900".
    for k in range(1, 9):
        if int(str(n)[2 * (k - 1)]) != k:
            return False
    return True

# Start with the root of the largest number
# 19293949596979899 -> 138,902,662
n = 1389026657
subtracting = 4
while not isValid(n ** 2):
    n -= subtracting
    # Alternate subtracting 4 and 6
    # to get numbers ending in 3 and 7
    if subtracting == 4:
        subtracting = 6
    else:
        subtracting = 4
print(f'{n * 10}^2 = {n * n * 100}')
```
Running the loop, we get
```
1389019170^2 = 1929374254627488900
0.0008701999999999877 seconds.
```
Thus, our unique number whose square follows the pattern is **1389019170**. The time is very quick because the answer is fairly close to the upper limit.