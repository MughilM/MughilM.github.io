---
layout: post
title: "#4 - Largest palindrome product"
date: 2015-07-23 12:06
number: 4
tags: [05_diff, Brute force]
---
> A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 x 99.
> 
> Find the largest palindrome made from the product of two 3-digit numbers.
{:.lead}
* * *

First, we need a function to determine if a number is a palindrome. Using Python, we can convert the number to a string, reverse it, and check if they are equal.
```python
# file: "problem004.py"
def isPalindrome(n):  
   num = str(n)  
   return num == num[::-1]
```
Next, the built-in method `itertools.combinations` allows to quickly iterate through all products of two 3-digit numbers. Because we want the largest, we will reverse the list of products after removing duplicates, and select the first palindrome we find.
```python
# file: "problem004.py"
numbers = sorted(set([a * b for a, b in combinations(range(100, 1000), r=2)]))  
  
i = -1  
while not isPalindrome(numbers[i]):  
   i -= 1  
  
print(numbers[i])
```
Running this short code gives us
```
906609
0.07893140008673072 seconds.
```
Therefore, **906609** is our answer.