---
layout: post
title: "#70 - Totient permutation"
date: 2019-08-14 12:37
number: 70
tags: [20_diff]
---
> Euler's Totient function, $\phi(n)$ [sometimes called the phi function], is used to determine the number of positive numbers less than or equal to $n$ which are relatively prime to $n$. For example, as 1, 2, 4, 5, 7, and 8, are all less than nine and relatively prime to nine, $\phi(9)=6$. The number 1 is considered to be relatively prime to every positive number, so $\phi(1)=1$.
>
> Interestingly, $\phi(87109)=79180$, and it can be seen that $87109$ is a permutation of $79180$.
>
> Find the value of $n, 1<n<10^7$, for which $\phi(n)$ is a permutation of $n$ and the ratio $n/\phi(n)$ produces a minimum.
{:.lead}
* * *

There are a couple of changes from [#69 - Totient maximum](/blog/project_euler/2017-06-21-069-Totient-maximum){:.heading.flip-title}. We are looking for the **minimum** ratio as opposed to the maximum. Our upper bound has also increased to 10 million. The approach we used in problem 69 will not be good enough with this increased limit.

If $n/\phi(n)$ needs to be minimuzed, then $\phi(n)$ needs to be maximized. The totient function is directly related by the distinct prime factors of $n$ through

$$
\phi(n) = n\prod_{p|n}\left(1-\frac{1}{p}\right)
$$

Notice that $\left(1-\frac{1}{p}\right)$ is something less than 1. Thus, the **more** prime factors we have, the **smaller** $\phi(n)$ will be. If $n$ is prime, then $\phi(n)=n-1$. However, $n-1$ will never be a permutation of $n$. The next case is when $n$ has 2 prime factors. If $n=p_1p_2$, then

$$
\begin{aligned}
	\phi(n)&=p_1p_2\left(1-\frac{1}{p_1}\right)\left(1-\frac{1}{p_2}\right)
	\\ &=
	p_1p_2\left(\frac{p_1-1}{p_1}\right)\left(\frac{p_2-1}{p_2}\right)
	\\ &=
	(p_1-1)(p_2-1)
\end{aligned}
$$

We want to keep this product as close to our upper bound of $10^7$ as possible, so we will need to choose prime factors close to $\sqrt{10^7}\approx 3162.278$. One way is to choose a "radius" of prime numbers around this value to check. There will need to be some trial and error, but I eventually settled on a radius of 2000. We generate primes between 1162 and 5162 and check to see when the ratio is minimized.
```python
# file: "problem070.py"
limit = 10 ** 7
center = int(limit ** 0.5)
primes = np.array(primesieve.primes(center - 2000, center + 2000))
# Keep a valid list of permutations...
totientPermutations = []
centerLoc = np.searchsorted(primes, center)
for i in range(centerLoc, len(primes)):
    # Find location of largest factor which will
    # not result in a product over limit. It's right-
    # justified, so subtract 1
    largestFactor = np.searchsorted(primes, limit // primes[i]) - 1
    # Going backwards, calculate phi(n), and n, and append
    # to our list of n is a permutation of phi(n)
    for j in range(largestFactor, -1, -1):
        # Phi(n) = (p1 - 1)(p2 - 1)
        phi = (primes[i] - 1) * (primes[j] - 1)
        n = primes[i] * primes[j]
        if sorted(str(n)) == sorted(str(phi)):
            totientPermutations.append([n, phi])

# Convert to numpy array, take ratios,
# and find the one with the smallest.
totientPermutations = np.array(totientPermutations)
ratios = totientPermutations[:, 0] / totientPermutations[:, 1]
print(totientPermutations[np.argmin(ratios)])
```
The output is 
```
[8319823 8313928]
0.19818319999999995 seconds.
```
Thus, $n=\mathbf{8319823}$ will gives us the minimum ratio. The totient is a permutation of $n$ as well.