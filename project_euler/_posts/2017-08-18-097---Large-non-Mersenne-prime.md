---
layout: post
title: "#97 - Large non-Mersenne prime"
date: 2017-08-18 01:10
number: 97
tags: [05_diff]
---
> The first known prime found to exceed one million digits was discovered in 1999, and is a Mersenne prime of the form $2^{6972593}-1$; it contains exactly 2,098,960 digits. Subsequently other Mersenne primes, of the form $2^p-1$, have been found which contain more digits.
> 
> However, in 2004 there was found a massive non-Mersenne prime which contains 2,357,207 digits: $28433\times2^{7830457}+1$.
> 
> Find the last ten digits of this prime number.
{:.lead}
* * *

We take the mod $10^{10}$ to find the last ten digits of a number. Calculate the number directly with Python:
```python
# file: "problem097.py"
print((28433 * 2 ** 7830457 + 1) % (10 ** 10))  # one line!!
```
We get an answer instantly,
```
8739992577
0.038738599978387356 seconds.
```
Therefore, the last ten digits are **8739992577**.
### Alternative method
If our language is unable to hold large numbers, then we will need to take an approach that was first demonstrated in [#48 - Self-powers](/blog/project_euler/2017-02-02-048-Self-powers){:.heading.flip-title}, where through the use of clever modular tricks, we calculate the value of 

$$
28433\times2^{7830457}+1\mod 10^{10}
$$
directly. Please see that problem for implementation details. We would first calculate $2^{7830457}\mod 10^{10}$ before multiplying and adding through. 