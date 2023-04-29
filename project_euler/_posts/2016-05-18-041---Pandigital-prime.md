---
layout: post
title: "#41 - Pandigital prime"
date: 2016-05-18 10:26
number: 41
tags: [05_diff]
---
> We shall say that an $n$-digit number is pandigital if it makes use of all the digits 1 to $n$ exactly once. For example, 2143 is a 4-digit pandigital and is also prime.
> 
> What is the largest $n$-digit pandigital prime that exists?
{:.lead}
* * *

We need a way to reduce the number of primes we need to check. Let's do some case-by-case analysis on the size of $n$.

If $n=9$, then it has 1, 2, ..., 9 in the number. The sum of these digits is 45, and recall that if the sum of the digits is 3 or 9, then the original number is divisible by 3 or 9. Thus, $\mathbf{n\neq 9}$.

The same analysis holds for $n=8$, the sum in this case is 36, which means any 8-digit pandigital number is divisible by 9. 

The process breaks for $n=7$, as this sum is 28. However, the pattern holds for both 6-digit (sum = 21) and 5-digit (sum = 15) pandigital numbers, which are all divisible by 3. 

Therefore, we will test all 7-digit primes, and hopefully we find a match. If we don't, then we would test 4-digit numbers on down. We will generate primes between 1000000 and 7654321, and check from the top. The first prime which is pandigital is our answer.
```python
# file: "problem041.py"
primes = primesieve.primes(1000000, 7654321)[::-1]
# Find the first prime (going from largest to smallest)
# that is pandigital.
sevenDigitPan = set('1234567')
for p in primes:
    if set(str(p)) == sevenDigitPan:
        print(p)
        break
```
The result is,
```
7652413
0.048801859630159886 seconds.
```
Therefore, largest pandigital prime is **7652413**.
