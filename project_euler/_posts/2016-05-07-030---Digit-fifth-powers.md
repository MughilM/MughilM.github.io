---
layout: post
title: "#30 - Digit fifth powers"
date: 2016-05-07 09:15
number: 30
tags: [05_diff]
---
> Surprisingly there are only three numbers that can be written as the sum of the fourth powers of their digits:
> 
> $$
> \begin{aligned}
> 1634 &= 1^4 + 6^4 + 3^4 + 4^4
> \\
> 8208 &= 8^4 + 2^4 + 0^4 + 8^4
> \\
> 9474 &= 9^4 + 4^4 + 7^4 + 4^4
> \end{aligned}
> $$
> 
> As $1=1^4$ is not a sum it is not included.
> 
> The sum of these numbers is 1634 + 8208 + 9474 = 19316.
> 
> Find the sum of all the numbers that can be written as the sum of fifth powers of their digits.
{:.lead}
* * *

We can do this with a simple for loop. However, we do not yet know the upper bound. Given $k$ digits, the largest number we can have is $k$ 9's. The sum of fifth powers of this number is therefore $k\times 9^5$. Notice that if $k=7$, the sum is $7\times 9^5 = 413\,343$, a six-digit number. Therefore, it is impossible to have a number that is 7 or more digits long be a sum of fifth powers of its digits. Additionally, if $k=6$, then the sum is $6\times 9^5 = 354\,294$. In that case, we don't need to check numbers bigger than this.

For calculating the sum of fifth powers, we can convert the number to a string and loop through as seen below.
```python
# file: "problem030.py"
limit = 6 * 9 ** 5 + 1
s = 0
for i in range(10, limit):
    if sum([int(digit) ** 5 for digit in str(i)]) == i:
        s = s + i
print(s)
```
The output after running is,
```
443839
1.061222499993164 seconds.
```
Therefore, **443839** is the sum of all numbers that can be written as the sum of the fifth powers of their digits.