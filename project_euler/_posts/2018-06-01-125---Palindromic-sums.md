---
layout: post
title: "#125 - Palindromic sums"
date: 2018-06-01 09:51
number: 125
tags: [25_diff]
---
> The palindromic number 595 is interesting because it can be written as the sum of consecutive squares: $6^2+7^2+8^2+9^2+10^2+11^2+12^2$.
>
> There are exactly eleven palindromes below one-thousand that can be written as consecutive square sums, and the sum of these palindromes is 4164. Note that $1^2=0^2+1^2$ has not been included as this problem is concerned with the squares of positive integers.
>
> Find the sum of all the numbers less than $10^8$ that are both palindromic and can be written as the sum of consecutive squares.
{:.lead}
* * *

We can do a brute force solution, where we loop through all possible consecutive square sums. The maximum number in this sum will be $\left(10^4\right)^2$. If we encounter a palindrome sum, then we add to a running list. We can also cut off the sum if we exceed the limit. We also want to `set()` the final list, to get rid of duplicates.
```python
# file: "problem125.py"
limit = 10 ** 8
squareLim = int(limit ** 0.5)
cumsumSquares = []
for i in range(1, squareLim + 1):
    s = i ** 2
    for j in range(i+1, squareLim + 1):
        s += j ** 2
        if s > limit:
            break
        if str(s) == str(s)[::-1] and s not in cumsumSquares:
            cumsumSquares.append(s)
print(sum(set(cumsumSquares)))
```
Running this short double for loop, we get
```
2906969179
0.49573599999999995 seconds.
```
Thus, the sum of all palindromes that can be written as a consecutive square sum is **2906969179**.