---
layout: post
title: "#16 - Power digit sum"
date: 2015-08-07 20:45
number: 16
tags: [05_diff]
---
> $2^{15} = 32768$ and the sum of its digits is $3 + 2 + 7 + 6 + 8 = 26$.
> 
> What is the sum of the digits of the number $2^{1000}$?
{:.lead}
* * *

As said in [#13 - Large sum](/blog/project_euler/2015-07-27-013-Large-sum){:.heading.flip-title}, Python is really good at handling large numbers. All we have to do here is compute teh large power, convert it to string, and add all the digits.
```python
# file: "problem016.py"
# Create array of digits in 2 ^ 1000
num = [int(i) for i in str(2 ** 1000)]
# Add them up
print(sum(num))
```
Running this short code,
```
1366
0.00012523456790123458 seconds.
```
Therefore, the sum of the digits is **1366**.