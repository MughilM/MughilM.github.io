---
layout: post
title: "#112 - Bouncy numbers"
date: 2019-02-11 12:01
number: 112
tags: [15_diff]
---
> Working from left-to-right if no digit is exceeded by the digit to its left it is called an increasing number; for example, 134468.
>
> Similarly if no digit is exceeded by the digit to its right it is called a decreasing number; for example, 66420.
>
> We shall call a positive integer that is neither increasing nor decreasing a "bouncy" number; for example, 155349.
>
> Clearly there cannot be any bouncy numbers below one-hundred, but just over half of the numbers below one-thousand (525) are bouncy. In fact, the least number for which the proportion of bouncy numbers first reaches 50% is 538.
>
> Surprisingly, bouncy numbers become more and more common and by the time we reach 21780 the proportion of bouncy numbers is equal to 90%.
>
> Find the least number for which the proportion of bouncy numbers is exactly 99%.
{:.lead}
* * *

We can do a brute force check to see if a number is bouncy. We can scan the number left to right, and find the first pair of numbers which does not follow the bouncy characteristic. In this way, we have a "fast-fail" approach where we don't have to search the whole number if it's bouncy.

For example, 155349 is bouncy. The first two digits are "15", and since $5>1$, all digits after that should be non-decreasing. However, "53" violates this, thus the number is bouncy. We will start from 21780, and count up until we reach a proportion of 99\%.
```python
# file: "problem112.py"
def isBouncy(x):
    # Convert to list of digits
    digits = [int(d) for d in str(x)]
    # Find the first distinct pair
    # and set whether the number should
    # be increasing or decreasing.
    i = 0
    while i < len(digits) - 1 and digits[i] == digits[i+1]:
        i += 1
    # If we've reached the end, then
    # all numbers are the same, so
    # it's not bouncy
    if i == len(digits) - 1:
        return False
    # Otherwise, find the direction
    if digits[i + 1] < digits[i]:
        increasing = False
    else:
        increasing = True
    # Now starting from the next pair,
    # see if the condition holds
    i += 1
    while i < len(digits) - 1:
        if (digits[i + 1] < digits[i] and increasing) or \
                (digits[i + 1] > digits[i] and not increasing):
            return True
        i += 1
    return False

bouncyNums = 21780 * 9 / 10
n = 21780

while bouncyNums / n < 0.99:
    n += 1
    if isBouncy(n):
        bouncyNums += 1

print('First reaches 99% at n =', n)
```
Running this short loop, we get
```
First reaches 99% at n = 1587000
4.4108225 seconds.
```
Thus, **1587000** is the first number which results in 99\% bouncy numbers below it.