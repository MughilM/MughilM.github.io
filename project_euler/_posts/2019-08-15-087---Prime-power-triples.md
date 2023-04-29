---
layout: post
title: "#87 - Prime power triples"
date: 2019-08-15 10:51
number: 87
tags: [20_diff]
---
> The smallest number expressible as  the sum of a prime square, prime cube, and prime fourth is 28. In fact, there are exactly four numbers below fifty that can be expressed in such a way:
>
> $$
> 28 = 2^2+2^3+2^4
> \\
> 33 = 3^2+2^3+2^4
> \\
> 49 = 5^2+2^3+2^4
> \\
> 47 = 2^2+3^3+2^4
> $$
>
> How many numbers below fifty million can be expressed as the sum of a prime square, prime cube, and prime fourth power?
{:.lead}
* * *

Due to the sum, the primes need to be less than $\sqrt{50000000}=7071.068$. We can easily generate all primes up to this limit, and subsequently choose a prime fourth power, third power, and second power on down.

Once we've chosen a prime to represent the fourth power, we subtract this from fifty million, and take the cube root to obtain the maximum prime that can be cubed.We do the same with the square root. With this method, we can greatly reduce the number of primes we have to test. We use a boolean sieve which we sum at the end in order to count the numbers.
```python
# file: "problem087.py"
def sieve(n):
    primes = list(range(2, n+1))
    # For each number, cross out
    # numbers that are multiples of it.
    for i, p in enumerate(primes):
        j = i + 1
        while j < len(primes):
            if primes[j] % p == 0:
                del primes[j]
            j += 1
    return primes

limit = 50000000
primes = sieve(int(limit ** 0.5) + 1)

sumSieve = [False] * limit
k = 0
while primes[k] ** 4 < limit:
    s1 = primes[k] ** 4
    i = 0
    while primes[i] < (limit - s1) ** (1/3):
        s2 = s1 + primes[i] ** 3
        j = 0
        while j < len(primes) and primes[j] < (limit - s2) ** (1/2):
            s3 = s2 + primes[j] ** 2
            sumSieve[s3] = True
            j += 1
        i += 1
    k += 1
print(sum(sumSieve))
```
Running our loop gets us an output of
```
1097343
4.7940836000000004 seconds.
```
Thus, the sum of all numbers that satisfy the condition is **1097343**.
