---
layout: post
title: "#124 - Ordered radicals"
date: 2019-06-12 18:45
number: 124
tags: [25_diff]
---
> The radical of $n$, $\text{rad}(n)$, is the product of the distinct prime factors of $n$. For example, $504=2^3\times3^2\times7$, so $\text{rad}(504)=2\times3\times7=42$.
>
> If we calculate $\text{rad}(n)$ for $1\leq n\leq 10$, then sort them on $\text{rad}(n)$, and sorting on $n$ if the radical values are equal we get:
>
> | Unsorted       |                              |      | Sorted         |                              |                |
> | -------------- | ---------------------------- | ---- | -------------- | ---------------------------- | -------------- |
> | $\mathbf{n}$ | $\textbf{rad}\mathbf{(n)}$ |      | $\mathbf{n}$ | $\textbf{rad}\mathbf{(n)}$ | $\mathbf{k}$ |
> | 1              | 1                            |      | 1              | 1                            | 1              |
> | 2              | 2                            |      | 2              | 2                            | 2              |
> | 3              | 3                            |      | 4              | 2                            | 3              |
> | 4              | 2                            |      | 8              | 2                            | 4              |
> | 5              | 5                            |      | 3              | 3                            | 5              |
> | 6              | 6                            |      | 9              | 3                            | 6              |
> | 7              | 7                            |      | 5              | 5                            | 7              |
> | 8              | 2                            |      | 6              | 6                            | 8              |
> | 9              | 3                            |      | 7              | 7                            | 9              |
> | 10             | 10                           |      | 10             | 10                           | 10             |
>
> Let $E(k)$ be the $k^{\text{th}}$ element in the sorted $n$ column; for example, $E(4) = 8$ and $E(6) = 9$.
>
> If $\text{rad}(n)$ is sorted for $1\leq n \leq 100000$, find $E(10000)$.
{:.lead}
* * *

Since we are sorting, we must calculate the radical for each $n$ up to 100000. A brute force solution calculates the prime factorization each time, but we can do better. The key point to realize is that **the radical stays constant as long as the distinct prime factors are the same.** For example, using the example in the problem, *any* number with 2, 3, and 7 as its distinct prime factors, will have a radical of 42. Since each integer has a unique prime factorization, we can loop through *all* possible distinct prime factor sets, then add all numbers stemming from those factor sets into an array (along with its constant radical) and finally sort it.
## Looping through factor sets
We use a recursive algorithm to loop through all factor sets of a certain size up to a limit. For example, let's say the set size was 3 and our limit was 100. The set with the smallest product is {2, 3, 5}, which is 30.  Now, keeping 2 and 3 constant, this means $2\times3\times p \leq 100 \rightarrow p \leq 16.67$. Thus, the only possibilities for the last factor is 5, 7, 11, and 13. Then, in the next loop, the second factor becomes 5, which leads to the last factor being no more than 10 (7 being the only satisfactory factor). To avoid duplicating the sets, we'll make sure the primes are increasing when going left to right.

Once this set size is finished, we move on to a set size of 4. We keep increasing the set size until multiplying the smallest $t$ prime numbers leads to exceeding the limit. In the 100 case, it is 3. For 1000, it is 4 (2, 3, 5, and 7).
```python
# file: "problem124.py"
def loopThroughFactorizations(numOfFacts, primes, limit):
    if numOfFacts == 1:
        i = 0
        while (i < len(primes)) and (primes[i] < limit):
            yield [primes[i]]
            i += 1
    else:
        upperBound = limit ** (1 / numOfFacts)
        i = 0
        while (i < len(primes)) and (primes[i] < upperBound):
            for subFactorization in loopThroughFactorizations(numOfFacts - 1, primes[i + 1:], limit / primes[i]):
                yield [primes[i]] + subFactorization
            i += 1
```
But this only gets the distinct prime factor set. We still need all integers with that set of distinct prime factors.
## Looping through all integers with specific distinct prime factors
This portion is also recursive. We increment the power on each prime, and calculate the maximum power of the remainign prime factors using logarithms. For example, with a limit of 1000 and the set of {2, 3, 7}, the maximum exponent 2 can have is $\lfloor \log_2\frac{1000}{3\times 7}\rfloor = \mathbf{5}$. Setting an exponent on 2 will effect the maximum exponents for the rest of the prime factors.
```python
# file: "problem124.py"
def loopThroughFactorPowers(factorization, limit):
    if len(factorization) == 1:
        base = factorization[0]
        powerLimit = math.log(limit, base)
        for power in range(1, math.ceil(powerLimit)):
            yield [power]
    else:
        prodOfRemaining = 1
        for number in factorization[1:]:
            prodOfRemaining *= number
        base = factorization[0]
        powerLimit = math.log(limit / prodOfRemaining, base)
        for power in range(1, math.ceil(powerLimit)):
            for remPowers in loopThroughFactorPowers(factorization[1:], limit / base ** power):
                yield [power] + remPowers
```
## Implementation
I use `primesieve.primes` to get my primes, and initialize my array with (1, 1). Passing in a list of tuples to the `sorted()` function will result in sorting the values by the first value, and then the second value, which we can take advantage of by placing the radical value first, then the integer.
```python
# file: "problem124.py"
limit = 100001
primes = primesieve.primes(limit)
rads = [(1, 1)]
# Find the maximum number of primes we can 
# multiply together
maxPrimes = 0
prod = 1
while prod < limit:
    prod *= primes[maxPrimes]
    maxPrimes += 1

for numOfFacts in range(1, maxPrimes):
    for factorization in loopThroughFactorizations(numOfFacts, primes, limit):
        rad = 1
        for factor in factorization:
            rad *= factor
        for factorPowers in loopThroughFactorPowers(factorization, limit):
            number = 1
            for factor, power in zip(factorization, factorPowers):
                number *= factor ** power
            rads.append((rad, number))
E = sorted(rads)
print(E[9999])
```
Running this code leads to an output of,
```
(1947, 21417)
0.41047860006801784 seconds.
```
Therefore, $E(10000)=\mathbf{21417}$, whose radical is 1947.