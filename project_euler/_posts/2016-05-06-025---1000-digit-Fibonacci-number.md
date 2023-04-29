---
layout: post
title: "#25 - 1000-digit Fibonacci number"
date: 2016-05-06
number: 25
tags: [05_diff]
---
> The Fibonnaci sequence is defined by the recurrence relation:
> 
> $$
> F_n = F_{n-1} + F_{n-2}
> $$
> 
> where $F_1=1$ and $F_2=1$.
> 
> Hence the first 12 terms will be:
> 
> $$
> \begin{aligned}
> 	F_1 &= 1 \\
> 	F_2 &= 1 \\
> 	F_3 &= 2 \\
> 	F_4 &= 3 \\
> 	F_5 &= 5 \\
> 	F_6 &= 8 \\
> 	F_7 &= 13 \\
> 	F_8 &= 21 \\
> 	F_9 &= 34 \\
> 	F_{10} &= 55 \\
> 	F_{11} &= 89 \\
> 	F_{12} &= 144
> \end{aligned}
> $$
> 
> The 12th term, $F_{12}$, is the first term to contain three digits.
>
> What is the index of the first term in the Fibonacci sequence to contain 1000 digits?
{:.lead}
* * *

To quickly figure out the number of digits of a number $n$, we do $\lfloor \log_{10} n \rfloor + 1$. This prevents us from converting to a string every time, which is a relatively expensive operation. Therefore, we can continually generate numbers until we find the number we want. 
```python
# file: "problem025.py"
digitNum = 1000
a = 1
b = 1
n = 2
while math.log10(b) + 1 < digitNum:
    temp = a + b
    a = b
    b = temp
    n += 1

print(n)
```
Running this quick loop,
```
4782
0.0020837000338360667 seconds.
```
Thus, the **4782**nd Fibonacci number has at least 1000 digits.