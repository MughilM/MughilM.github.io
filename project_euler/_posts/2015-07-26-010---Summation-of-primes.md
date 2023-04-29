---
layout: post
title: "#10 - Summation of primes"
date: 2015-07-26 19:29
number: 10
tags: [05_diff]
---
> The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
> 
> Find the sum of all primes below two million.
{:.lead}
* * * 

Since finding primes is the main crux of the problem, I will refran from using the `primesieve` package. In this case, I will use the same sieve we generated from [#7 - 10001st prime](/blog/project_euler/2015-07-26-007-10001st-prime){:.heading.flip-title}. This time we need to go until two million. To get the actual values, we will just use `enumerate`, which pairs the index with the actual value.
```python
# file: "problem010.py"
limit = 2000000
sieve = [True] * (limit + 1)
sieve[0] = False
sieve[1] = False
# Start at index 2, so the index = prime number
p = 2
while p < limit + 1:
    # If it's composite, then go to the next one
    if not sieve[p]:
        p += 1
        continue
    # Mark all multiples
    for i in range(2 * p, limit + 1, p):
        sieve[i] = False
    p += 1

s = 0
for i, mark in enumerate(sieve):
    if mark:
        s += i
print(s)
```
Running the loop,
```
142913828922
1.7028809000039473 seconds.
```
Therefore, the sum of all primes under two million is **142913828922**.