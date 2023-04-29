---
layout: post
title: "#21 - Amicable numbers"
date: 2015-08-13 10:21
number: 21
tags: [05_diff]
---
> Let $d(n)$ be defined as the sum of proper divisors of $n$ (numbers less than $n$ which divide evenly into $n$).
> If $d(a) = b$ and $d(b) = a$, where $a\neq b$, then $a$ and $b$ are an amicable pair and each of $a$ and $b$ are called amicable numbers.
> 
> For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore $d(220) = 284$. The proper divisors of 284 are 1, 2, 4, 71, and 142; so $d(284) = 220$.
> 
> Evaluate the sum of all the amicable numbers under 10000.
{:.lead}
* * *

We will need an efficient method to finding the sum of the proper divisors of $n$ outside of a brute force approach. Using the prime factorization of $n$, it is possible to directly derive the sum of the proper divisors, without explicitly finding the divisors themselves. Let's use the number in the example:

$$
220 = 2^2\times 5\times 11
$$
The divisors of $n$ will be a subproduct of the prime factorization of $n$. For example, 10 is a divisor, and $10=2\times 5$. Below is the full list:

$$
1 \\ 2 \\ 2^2 \\ 5 \\ 2\times 5 \\ 11 \\ 2^2\times 5 \\ 2\times 11 \\ 2^2\times 11 \\ 5\times 11 \\ 2\times 5\times 11 \\ 2^2\times 5\times 11
$$

Writing this another way, we can see a pattern,

$$
2^0 5^0 11^0 \\
2^1 5^0 11^0 \\
2^2 5^0 11^0 \\
2^0 5^1 11^0 \\
2^1 5^1 11^0 \\
2^0 5^0 11^1 \\
2^2 5^1 11^0 \\
2^1 5^0 11^1 \\
2^2 5^0 11^1 \\
2^0 5^1 11^1 \\
2^1 5^1 11^1 \\
2^2 5^1 11^1
$$

If we simplify the sum, we can easily factor as follows:

$$
\begin{aligned}
2^0 5^0 11^0 + 2^1 5^0 11^0 + \cdots + 2^2 5^1 11^1 &=
	(2^0 + 2^1 + 2^1)(5^0 11^0 + 5^1 11^0 + 5^0 11^1 + 5^1 11^1)
\\ &=
(2^0 + 2^1 + 2^2)(5^0 + 5^1)(11^0 + 11^1)
\end{aligned}
$$

This sum includes $n$ itself, so we need to subtract it off. Additionally, notice that each individual sum in the product is a sum of a geometric series. Recall that if $a$ is the first term and $r$ is the common ratio, then

$$
\sum_{i=0}^n ar^i = a\left( \frac{r^{n+1} - 1}{r - 1} \right)
$$

In our case, $a = 1$. Therefore, if we have $m$ prime factors $p_1, p_2, \cdots, p_m$ with exponents $k_1, k_2, \cdots, k_m$ respectively, then 

$$
d(n) = \prod_{i=1}^m \frac{p_i^{k_i+1}}{p_i-1} - n
$$

In our example, since $220 = 2^2\times 5\times 11$, this means

$$
d(220) = \left( \frac{2^3 - 1}{2 - 1} \right)\left( \frac{5^2 - 1}{5 - 1} \right)\left( \frac{11^2 - 1}{11 - 1} \right) - 220 = 7(6)(12) - 220 = \boxed{284}
$$

Below is the Python function `d(n, primes)` which does the same calculation.
```python
# file: "problem021.py"
def d(n, primes):
    # 0 and 1 have no proper divisors, so return 0.
    if n <= 1:
        return 0
    # If n is a prime, then by definition d(n) = 1
    if n in primes:
        return 1
    # underPrimes = primes[primes <= n ** 0.5]
    primeFacts = []
    powers = []
    ### PRIME FACTORIZATION
    # Test each prime to see if it divides
    num = n
    for p in primes:
        # Keep dividing until it can't.
        # Keep a counter for the number of times.
        power = 0
        while num % p == 0:
            num //= p
            power += 1
        # If we've divided, then this is a factor.
        if power > 0:
            primeFacts.append(p)
            powers.append(power)
        # If num became 1, then we've exhausted all
        # factors, no need to check.
        if num == 1:
            break
    ### CALCULATING SUM OF PROPER DIVISORS
    # Take each prime and respective power,
    # and do (p^(k+1) - 1)/(p-1)
    s = 1
    for prime, power in zip(primeFacts, powers):
        s *= (prime ** (power + 1) - 1) // (prime - 1)
    # The product includes n, which isn't a
    # PROPER divisor, so subtract.
    s -= n
    return s
```
Once we have this, we just loop through all numbers up to 10000, and see if we find amicable pairs.
```python
# file: "problem021.py"
limit = 10000
primes = primesieve.primes(limit)
# For each number from 0 to 9999, find
# d(n).
Ds = [d(i, primes) for i in range(limit)]
# Now go through the list and search
# for amicable numbers.
s = 0
for n, propSum in enumerate(Ds):
    # The first check is necessary because
    # we don't want perfect numbers.
    if n != propSum and propSum < limit and Ds[propSum] == n:
        s += n
        print(n, end=' ')
print()
print(s)
```
Running our code results in an output of,
```
220 284 1184 1210 2620 2924 5020 5564 6232 6368 
31626
0.13923039997462183 seconds.
```
Thus, **31626** is the sum we are looking for. You can also see the other pairs we have found.