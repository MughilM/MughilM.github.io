---
layout: post
title: "#36 - Double-base palindromes"
date: 2016-05-10 11:56
number: 36
tags: [05_diff]
---
> The decimal number, $585=1001001001_2$ (binary), is palindromic in both bases.
> 
> Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2.
> 
> (Please note that the palindromic number, in either base, may not include leading zeros.)
{:.lead}
* * *

Python has a convenient built-in function `bin()` that converts the number to binary. To check if a number is a palindrome, we will reverse and check for equality. 
```python
# file: "problem036.py"
s = 0
isPalindrome = lambda x: x == x[::-1]
for i in range(1, 1000000):
    if isPalindrome(str(i)) and isPalindrome(bin(i)[2:]):
        s += i
print(s)
```
The output of the code is,
```
872187
0.49978133307417255 seconds.
```
Thus, our needed sum is **872187**.