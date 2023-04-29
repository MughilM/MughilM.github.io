---
layout: post
title: "#123 - Prime square remainders"
date: 2017-07-06 11:03
number: 123
tags: [30_diff]
---
> Let $p_n$ be the $n^{\text{th}}$ prime: 2, 3, 5, 7, 11, ..., and let $r$ be the remainder when $(p_n-1)^n + (p_n+1)^n$ is divided by $p_n^2$.
> 
> For example, when $n=3$, $p_3=5$, and $4^3 + 6^3 = 280 \equiv 5\mod 25$.
> 
> The least value of $n$ for which the remainder first exceeds $10^9$ is 7037.
>
> Find the least value of $n$ for which the remainder first exceeds $10^{10}$.
{:.lead}
* * *

This is a straightforward problem. The numbers will start to get large beacuse they are raised to the power $n$, so it will be inefficient to directly calculate this value. Instead, we can use the modular approach we first saw in [#48 - Self-powers](/blog/project_euler/2017-02-02-048-Self-powers){:.heading.flip-title}. Outside of that, we just loop up until we get the satisfied value. I use the `primesieve` package once again.
```python
# file: "problem0123.py"
def mod(a, b, c):
    bbinary = '{0:b}'.format(b)
    modPowers = np.zeros(len(bbinary), dtype=object)
    # Calculate a^k mod c for all powers < b
    modPowers[0] = a % c  # a^1 mod c
    for i in range(1, len(modPowers)):
        modPowers[i] = (modPowers[i - 1] * modPowers[i - 1]) % c
    # Filter out the powers we don't need and multiply them
    return np.prod(modPowers[[x == '1' for x in bbinary[::-1]]]) % c

# The first p_n that exceeds 100,000 is n = 9593
n = 9593
while True:
    prime = nth_prime(n)
    r = ((mod(prime - 1, n, prime ** 2) + mod(prime + 1, n, prime ** 2)) % prime ** 2)
    if r > 10 ** 10:
        print(r)
        print(n)
        print(prime)
        break
    n += 1
```
Running this short program get us,
```
10001595590
21035
237737
1.0571725 seconds.
```
Thus, the least value for which we get a remainder above ten billion is the **21035th** prime, which is 237737.