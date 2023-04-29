---
layout: post
title: "#49 - Prime permutations"
date: 2017-02-02 09:53
number: 49
tags: [05_diff]
---
> The arithmetic sequence, 1487, 4817, 8147, in which each of the terms increases by 3330, is unusual in two ways: (i) each of the three terms are prime, and, (ii) each of the 4-digit numbers of are permutations of one another.
> 
> There are no arithmetic sequences made up 1-, 2-, or 3-digit primes, exhibiting this property, but there is one other 4-digit increasing sequence.
> 
> What 12-digit number do you form by concatenating the three terms in this sequence?
{:.lead}
* * *

We can use a double for loop instead of a triple for loop. For each pair of primes, we check if they are permutations of each other, by converting them to sorted strings and checking for equality.
```python
# file: "problem049.py"
numbers = {}
primes = primesieve.primes(10 ** 3, 10 ** 4)
for i in range(len(primes)):
    for j in range(i + 1, len(primes)):
        prime1 = primes[i]
        prime2 = primes[j]
        # check if a permutation
        if sorted(str(prime1)) == sorted(str(prime2)):
            diff = prime2 - prime1
            prime3 = prime2 + diff
            if sorted(str(prime2)) == sorted(str(prime3)) and prime3 in primes:
                print(prime1, prime2, prime3)
```
Running gives us the triplets we need:
```
1487 4817 8147
2969 6299 9629
0.9628449275774557 seconds.
```
Concatenating the triplet that's not in the problem is **296962999629**.