---
layout: post
title: "#26 - Reciprocal cycles"
date: 2016-05-06 13:09
number: 26
tags: [05_diff]
---
> A unit fraction contains 1 in the numerator. The decimal representation of the unit fractions with denominators 2 to 10 are given:
> 
> $$
> \begin{aligned}
> \frac{1}{2} &= 0.5 \\
> \frac{1}{3} &= 0.\overline{3} \\
> \frac{1}{4} &= 0.25 \\
> \frac{1}{5} &= 0.2 \\
> \frac{1}{6} &= 0.1\overline{6} \\
> \frac{1}{7} &= 0.\overline{142857} \\
> \frac{1}{8} &= 0.125 \\
> \frac{1}{9} &= 0.\overline{1} \\
> \frac{1}{10} &= 0.1
> \end{aligned}
> $$
> 
> where $0.1\overline{6}$ means 0.166666..., and has a 1-digit recurring cycle. It can be seen that $\frac{1}{7}$ has a 6-digit recurring cycle.
> 
> Find the value of $d<1000$ for which $\frac{1}{d}$ contains the longest recurring cycle in its decimal fraction part.
{:.lead}
* * *

There are 2 things to unpack. First, we need to detect if we have a cycle. Second, and relatedly, we need to determine the length of that cycle. With the compounding inaccuracies of floating-point divisions, we need a way to find a way to determine the cycle length from $d$ itself.

[This answer on Quora](https://www.quora.com/What-determines-the-number-of-digits-for-recurring-decimals) and the [Wikipedia](https://en.wikipedia.org/wiki/Repeating_decimal) article give us insight in determining the cycle length. Given our fraction $\frac{1}{d}$, we repeatedly multiply by 10 until $gcd(10,d) = 1$. Every time we multiply, we must reduce the fraction to its lowest terms. Multiplying by 10 has the same effect as moving the decimal point to the right. Therefore, after $k$ steps, the original fraction and the multipled fraction will have the same decimal repeating cycle. To find $k$, we find the smallest $k$ that satisfies $10^k\equiv 1\mod d$. Let's do an example with $\frac{1}{55}$.

Observe that $gcd(10,55)=5\neq 1$. Thus, multiply the fraction to get $\frac{10}{55} = \frac{2}{11}$. Now $gcd(10,11) = 1$, our condition is satisfied and we can move to the next step. We are looking for the smallest $k$ that satisfies $10^k\equiv 1\mod 11$.

$$
\begin{aligned}
10^1 &\equiv 10\mod 11 = 10 \\
10^2 &\equiv 100\mod 11 = 1
\end{aligned}
$$

Therefore, $k=2$ and we conclude that $\frac{1}{55}$ has a **2-digit** recurring cycle, and indeed it does, as the decimal expansion is $0.0\overline{18}$. 

What if during the first step we encounter a value of 1 for $d$?. This means that the decimal will not repeat, as the denominator only consisted of factors of 2 and 5. All fractions whose denominators with only these two prime factors will eventually terminate. Some intuition is that $\frac{1}{2}$ and $\frac{1}{5}$ are the only fractions with prime denominators that don't repeat. 

Let's revisit the [Wikipedia article](https://en.wikipedia.org/wiki/Repeating_decimal). The article states a **prime denominator** $p$ will have a recurring cycle length of $p-1$ _unless_ the number 10 is a so-called multiplicative root of $p$, in which case the length will be shorter. For the purposes of this problem, this fact is out-of-scope, and this check is not necessary, as we are looking for the longest cycle. **However, now we only need to check the prime numbers under 1000!** To efficiently find the greatest common divisor, we can use the [Euclidean algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm).
```python
# file: "problem026.py"
def euclidGCD(a, b):
    if a == 0:
        return b
    if a < b:
        return euclidGCD(b, a)
    return euclidGCD(a - b * (a//b), b)

maxK = 1
maxN = 11
for n in primesieve.primes(1000):
    numerator = 1
    denominator = n
    # While the gcd of the denominator
    # and 10 isn't 1...keep multiplying
    # by 10.
    while euclidGCD(denominator, 10) != 1:
        # Multiply numerator by 10,
        # and reduce.
        numerator *= 10
        gcd = euclidGCD(numerator, denominator)
        numerator //= gcd
        denominator //= gcd
    # If we ever encounter a denominator
    # of 1, that means the decimal doesn't
    # repeat, and we can skip this. Fun Fact,
    # this happens with numbers with only
    # 2 and 5 as their prime factors.
    if denominator == 1:
        continue
    # Once 10 is co-prime, find when
    # 10^k mod denominator is 1.
    # This k is the length of the recurrence.
    k = 1
    while (10 ** k) % denominator != 1:
        k += 1
    # Update if it's the biggest we found.
    if maxK < k:
        maxK = k
        maxN = n

print('d =', maxN, 'with', maxK, 'recurring digits.')
```
Running our code results in
```
d = 983 with 982 recurring digits.
0.0507964999997057 seconds.
```
Therefore, $\frac{1}{983}$ has the longest recurring decimal cycle at 982 digits.