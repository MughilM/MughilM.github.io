---
layout: post
title: "#46 - Goldbach's other conjecture"
date: 2017-01-31 19:47
number: 46
tags: [05_diff]
---
> It was proposed by Christian Goldbach that every odd composite number that can be written as the sum of a prime and twice a square.
> 
> $$
> \begin{aligned}
> 9 &= 7 + 2\times 1^2
> \\
> 15 &= 7 + 2\times 2^2
> \\
> 21 &= 3 + 2\times 3^2
> \\
> 25 &= 7 + 2\times 3^2
> \\
> 27 &= 19 + 2\times 2^2
> \\
> 33 &= 31 + 2\times 1^2
> \end{aligned}
> $$
> 
> It turns out that the conjecture was false.
> 
> What is the smallest odd composite that cannot be written as the sum of a prime and twice a square?
{:.lead}
* * *

A simple double for loop will work. Unlike before, we actually build the prime list as we go, to minimize running time. Given our starting number, we keep subtracting twice squares from it until it results in a prime. Then, we move to the next number. I use generator iterables in the code to simplify things.
```python
# file: "problem046.py"
n = 5
primes = []

while True:
    # Check if the number is prime
    if all(n % p for p in primes):
        primes.append(n)
    # Check if twice a square being subtracted
    # from this number results in a prime.
    else:
        if not any((n - 2 * k * k) in primes for k in range(1, int((n / 2) ** 0.5) + 1)):
            break
    n += 2

print(n)
```
Running the program results in an output of,
```
5777
0.08932956603362126 seconds.
```
Therefore, **5777** is the first odd composite number that can't be written as the sum of a prime and twice a square.