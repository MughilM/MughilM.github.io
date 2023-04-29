---
layout: post
title: "#225 - Tribonacci non-divisors"
date: 2017-08-08 09:52
number: 225
tags: [45_diff]
---
> The sequence 1, 1, 1, 3, 5, 9, 17, 31, 57, 105, 193, 355, 653, 1201 ... is defined by $T_1=1, T_2=1, T_3=1$ and $T_n=T_{n-1}+T_{n-2}+T_{n-3}$.
>
> It can be shown that 27 does not divide any terms of this sequence. In fact, 27 is the first odd number with this property.
>
> Find the 124th odd number that does not divide any number of this sequence.
{:.lead}
* * *

If $a$ is divisible by $b$, then $a\equiv 0\mod b$. This sequence is very closely related to the Fibonacci series, which adds only the previous two. On the subject of Fibonacci numbers and modular arithmetic, there is the [**Pisano period**](https://en.wikipedia.org/wiki/Pisano_period). If you were to take $F_n\mod k$ for any integer $k$, then eventually the sequences of remainders will repeat. The length of this repeating sequence (a.k.a period) is the Pisano period. 

Since the Tribonacci series is built the same way, we will get the same repeating sequence. So when we test to see if a number divides the sequence, we only need to test the values in the period.

When does the sequence repeat. The first 3 values are all ones, which are less than any number we test. Therefore, the modular of this will still be 1. After the first 3 ones, **if we encounter 3 ones again, then the sequence has repeated.**

If we encounter a 0 anywhere in the sequence, then we stop checking and move to the next odd number.
```python
# file: "problem225.py"
oddNums = 0
n = 27
while oddNums < 124:
    vals = [1,1,1,3]
    while vals[-3:] != [1,1,1] and vals[-1] != 0:
        vals.append(sum(vals[-3:]) % n)
    # Check if it's 111 or 0
    if vals[-1] != 0:
        oddNums += 1
    n += 2
# We went two over, so subtract 2...
print(n - 2)
```
Running our loop, we get
```
2009
2.212874299963005 seconds.
```
Therefore, the 124th odd number is **2009**.