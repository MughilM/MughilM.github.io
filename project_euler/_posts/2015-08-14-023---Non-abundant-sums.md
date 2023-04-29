---
layout: post
title: "#23 - Non-abundant sums"
date: 2015-08-14 09:10
number: 23
tags: [05_diff]
---
> A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.
> 
> A number $n$ is called deficient if the sum of its proper divisors is less than $n$ and it is called abundant if this sum exceeds $n$.
>
> As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number that cannot be expressed as the sum of two abundant numbers is less than this limit.
> 
> Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.
{:.lead}
* * *

The $d(n)$ function we developed in [#21 - Amicable numbers](/blog/project_euler/2015-08-13-021-Amicable-numbers){:.heading.flip-title}. To recap, given the prime factorization 

$$
n = p_1^{k_1}p_2^{k_2}\cdots p_m^{k_m}
$$

the sum of proper divisors is given by

$$
d(n) = \prod_{i=1}^m \frac{p_i^{k_i+1}}{p_i-1} - n
$$

In this problem, we have to go until 28123 and grab all numbers which are abundant. Additionally, it is **easier** to find the inverse of the statement: Find which numbers **can** be written as the sum of two abundant numbers.

Since the list is sorted, we can run a double for loop that goes through each pair of numbers. We can break out of the inner loop once we reach a sum bigger than 28123 to save some time. Once we have the numbers, we can sum these up, and subtract it from the sum of all numbers from 1 to 28123, which recall has a closed-from formula of $\frac{n(n+1)}{2}$.
```python
# file: "problem023.py"
limit = 28123
primes = primesieve.primes(limit)
Ds = [d(i, primes) for i in range(limit + 1) if d]

# Go through and remove all numbers which aren't
# abundant.
abundantNums = [i for i, num in enumerate(Ds) if num > i]

# Find all numbers which can be written
sums = set()
for i in range(len(abundantNums) - 1):
    for j in range(i, len(abundantNums)):
        s = abundantNums[i] + abundantNums[j]
        # Break out if s > 28123
        if s > limit:
            break
        # Add the sum to the list
        sums.add(s)
# We can subtract the sum of all
# numbers from 1 to 28123 from the sum
# of this set...
print(limit * (limit + 1) // 2 - sum(sums))
```
We use `set()` because we might encounter numbers which have two separate sums. Running the code above results in an output of
```
4179871
3.11050750000868 seconds.
```
Thus, **4179871** is the sum of all positive integers which can't be written as the sum of two abundant numbers.