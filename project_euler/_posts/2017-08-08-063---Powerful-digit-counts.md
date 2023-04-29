---
layout: post
title: "#63 - Powerful digit counts"
date: 2017-08-08 12:09
number: 63
tags: [05_diff]
---
> The 5-digit number, $16807=7^5$, is also a fifth power. Similarly, the 9-digit number, $134\,217\,728=8^9$, is a ninth power.
> 
> How many $n$-digit positive integers exist which are also an $n$th power?
{:.lead}
* * *

First, we need to find some bounds for the solution to this problem. An $n$th power $a^n$ can only be an $n$-digit number if

$$
10^{n-1}<a^n<10^n
$$
We immediately see that $a<10$. For $n$, we can solve for it using the left half of the inequality:

$$
\begin{aligned}
10^{n-1} &< a^n
\\
\log_{10}\left(10^{n-1}\right) &< \log_{10}\left(a^n\right)
\\
n-1 &< n\log_{10}a
\\
n-n\log_{10}a &< 1
\\
n &< \frac{1}{1-\log_{10}a}
\end{aligned}
$$

The last expression is our upper bound for a given $a$. To check the length, the easiset way is to convert the number to a string.
```python
# file: "problem063.py"
count = 0
for a in range(2, 10):
    n = 1
    # Bound for the power
    while n <= 1/(1 - math.log10(a)):
        if len(str(a ** n)) == n:
            count += 1
        n += 1

print(count + 1) # for 1 ^ 1 = 1
```
Running this short gets an answer immediately,
```
49
7.499998901039362e-05 seconds.
```
Thus, there are **49** $n$-digit integers that are also $n$th powers.