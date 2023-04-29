---
layout: post
title: "#193 - Squarefree Numbers"
date: 2019-07-11 12:47
number: 193
tags: [55_diff]
---
> A positive integer $n$ is called squarefree, if no square of a prime divides $n$, thus 1, 2, 3, 5, 6, 7, 10, 11 are squarefree, but not 4, 8, 9, 12.
>
> How many squarefree numbers are there below $2^{50}$?
{:.lead}
* * *

The opposite of squarefree is **squareful**. There are fewer squareful numbers than squarefree, so we'll counting the squareful numbers and subtracting from the limit. The limit of $2^{50}$ is too big to create a bit array. However, we are only concerned numbers which contain squares of primes, so we only need primes up to the square root of the limit, or $2^{25}$ in this case.

To count the number of squareful numbers, we use the principle of inclusion/exclusion, as follows.
1. **Add** the numbers that contain a single square prime i.e. multiples of 4, 9, 16, etc.
2. When we do step 1, we double-added numbers that had **two** square prime factors (multiples of 36, 64, etc.), so we **subtract** these numbers. 
3. We double-subtracted the numbers that had **three** square prime factors (900 is the smallest such number), so we **add** these back in.
4. ...Continue alternating adding and subtracting numbers with 4, 5, … square prime factors, until the limit is crossed.

Given the upper limit $L$ and the product $P$, the number of multiples is simply $\lfloor L/P^2\rfloor$, which we will use to prevent looping through all the multiples, and instead only worry about the distinct square prime factors.

We need a list of primes up till $\sqrt{L}$, and we also need to know the maximum number of square primes we can multiply before crossing this limit, which can easily be calculated. In our case, this is 8.

To handle our inclusion/exclusion counting, I use a recursive function. This keeps track of $L$, the number of factors left $x$, the current product $P$, and the list of primes $p_1, p_2,\dots$. The base case is when $x=1$, and we yield the current product multiplied with each prime factor.

Otherwise, we loop through each prime and do the following steps:
1. We keep looping until the current product multiplied by the next $x$ factors exceeds the limit.
2. For each chosen prime $p_i$, the prime list is cut down to be in the range $[p_i+1,L/(P\times p_i)]$. The limit stays the same, since this is needed by each recursive call. The number of factors goes down by 1, the current product is updated through the multiplication with $p_i$.

To add the products when $x$ is odd, and subtract if $x$ is even, we multiply a $-1^{x+1}$ factor. This code counts the squareful numbers starting from 2, so we subtract from the limit, and account for 1 being squarefree.
```python
# file: "problem193.py"
def genProducts(primes, limit, factors=1, currProd=1):
    if factors == 1:
        for prime in primes:
            yield currProd * prime
        return
    i = 0
    while currProd * np.prod(primes[i:i + factors]) < limit:
        prime = primes[i]
        for product in genProducts(primesieve.primes(prime + 1, limit / (currProd * prime)), limit=limit,
                                   factors=factors - 1,
                                   currProd=currProd * prime):
            yield product
        i += 1


limit = 2 ** 50
primes = primesieve.primes(limit ** 0.5)
squareFulNums = 0
# Find maximum number of factors...
i = 1
prod = 2
while prod <= limit ** 0.5:
    prod *= primes[i]
    i += 1
maxFactors = i - 1

for numOfFactors in range(1, maxFactors + 1):
    totalProdCount = 0
    for product in genProducts(primes=primes, limit=limit ** 0.5, factors=numOfFactors, currProd=1):
        totalProdCount += (limit // product ** 2)
    squareFulNums += (-1) ** (numOfFactors + 1) * totalProdCount

print(limit - squareFulNums + 1)
```
Our output is thus,
```
684465067343070
12.7108755 seconds.
```
Therefore, the number of squarefree numbers under $2^{50}$ is **684465067343070**.