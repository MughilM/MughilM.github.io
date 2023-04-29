---
layout: post
title: "#5 - Smallest multiple"
date: 2015-07-23 12:36
number: 5
tags: [05_diff, Analytical solution]
---
> 2520 is the smallest number that can be divided by each of the numbers by 1 to 10 without any remainder.
> 
> What is the smallest positive number that is evenly divisible by all the numbers from 1 to 20?
{:.lead}
* * *

We need to find the Least Common Multiple (LCM) of all the numbers from 1 to 20. If we have the prime factorization of each number, then finding the LCM is simple.

To find the LCM of a list of numbers, we would union the prime factorizations of all the numbers together. This way, lower numbers get absorbed into higher factorizations. We then multiply all the unioned factorizations together. Here is an example with 1-10:

* $\mathbf{2} = 2$
* $\mathbf{3} = 3$
* $\mathbf{4} = 2^2$
* $\mathbf{5} = 5$
* $\mathbf{6} = 2\times3$
* $\mathbf{7} = 7$
* $\mathbf{8} = 2^3$
* $\mathbf{9} = 3^2$
* $\mathbf{10} = 2\times5$

We have four unique factors: 2, 3, 5, 7. The union of each factor is $2^3, 3^2, 5$, and $7$ respectively. Multiplying these together gets us the LCM:

$$
LCM(1, 2, 3, \dots, 10) = 2^3\times 3^2\times 5\times 7 = \mathbf{2520}
$$

However, there is one optimization we can do. Becuase we are doing consecutive numbers, all we need to find is the **highest power of each prime less than the limit**. There will be eventually be a prime where every power prime greater than that will just be the prime itself. For 1-10, that prime was 3. For 1-20, that prime will also be 5, since $5^2=25>20$. Therefore, we can calculate this product directly:

$$
LCM(1,2,3,\dots,20) = 2^4\times 3^2\times 5\times 7\times 11\times 13\times 17\times 19 = \mathbf{232792560}
$$
Therefore, **232792560** is our answer. If we wanted to actually code this, all we need is the `primesieve` package to get a list of primes less than the limit. We test $\log_p L$ for each prime $p$. As soon as this is less than 2, we can stop our loop and just multiply the rest of the prime numbers.

```python
# file; "problem005.py"
limit = 20  
prime_list = primes(limit)  
  
# Keep looping until log_p is less than 2  
prod = 1  
i = 0  
while np.emath.logn(prime_list[i], limit) > 2:  
   prod *= prime_list[i] ** int(np.emath.logn(prime_list[i], limit))  
   i += 1  
# Multiply the rest starting at i  
prod *= np.prod(prime_list[i:])  
  
print(prod)
```
Running the code gives,

```
232792560
0.00041200011037290096 seconds.
```
which is the same answer as the analytical method.
