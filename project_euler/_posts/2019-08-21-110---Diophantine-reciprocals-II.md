---
layout: post
title: "#110 - Diophantine reciprocals II"
date: 2019-08-21 15:17
number: 110
tags: [40_diff]
---
> In the following equation, $x$, $y$, and $n$ are positive integers.
> 
> $$
> \frac{1}{x} + \frac{1}{y} = \frac{1}{n}
> $$
> 
> It can be verified that when $n=1260$ there are 113 distinct solutions and this is the least value of $n$ for which the total number of distinct solutions exceeds one hundred.
> 
> What is the least value of $n$ for which the number of distinct solutions exceeds four million?
> 
> This problem is a much more difficult version of [#108 - Diophantine reciprocals I](/blog/project_euler/2019-08-20-108-Diophantine-reciprocals-I){:.heading.flip-title}
> {:.note}
{:.lead}
* * *

Please read the solution of [#108 - Diophantine reciprocals I](/blog/project_euler/2019-08-20-108-Diophantine-reciprocals-I){:.heading.flip-title} as it contains important concepts. In that problem, we assumed that the number of solutions will only have 3 and 5 as prime factors in order to keep the exponents of $n$ small. However, we cannot make the same assumption in this problem, as the number of solutions is much too large.

We need the least value of $n$ such that $f(n^2) > 8000000$. While we can't say anything about which prime factors are in this value, we _can_ say that the maximum prime factor is 47, since 3 is the minimum exponent, and $\lceil\log_3 8000000\rceil=15$, and the 15th prime factor is 47.

Unfortunately, while we can write an algorithm to generate all numbers with prime factors 2 to 47, it would take a massive amount of time to run, since we would need to run through all possible cobinations of factors (of which there are $2^{15}$) of them.

Instead, what we can do instead is find a **list of candidate solutions** and pick out our solution from that list. However, we need an upper bound in order to do that.

## Finding an upper bound
We need an upper bound $B$ such that there is some $n<B$ with at least 4 million solutions. We can use the _factorial_ function here, as it's one of the most fastest growing functions. We can find the least $K$ such that $K!$ has at least 4 million solutions. Recall we need the prime factorization of $K!$ in order to find the number of solutions. [Legendre's formula](https://en.wikipedia.org/wiki/Legendre's_formula), which utilizes divisibility to find the pwoer of each prime, can help us with this.

Once we've found the prime factorization of $K!$, we know the minimum solution must exist below it. At this point, we can loop through all possible values of the exponents, since we have the maximum prime number possible (47) and we have limits on each individual exponent.

This is done with recursion, where we can also decrease the remaining upper bound as we go e.g. if we have $2^8$ in our product so far, then the rest of the product shouldn't exceed $\frac{K!}{2^8}$. Our final roadmap is as follows:
1. Find the least integer $K$ such that $K!$ has at least 4 million solutions, or 8 million factors.
2. Of the prime factors in that factorization of $K!$, loop through all integers that are less than $K!$.
3. Any integers that have at least 8 million factors get added to a running list.
4. The minimum value of that list is our answer.

## Legendre's formula
Please see [this wiki](https://en.wikipedia.org/wiki/Legendre's_formula) for a detailed statement of the formula, as well as an example of it in action.
```python
# file: "problem110.py"
def primeFactorFactorial(n):
    # Use Legendre's formula.
    # We check until the greatest prime less than n
    primes = primesieve.primes(n)
    powers = []
    # For each prime...until halfway
    i = 0
    while primes[i] <= n/2:
        prime = primes[i]
        # Find largest power of prime
        # less than n...
        stopInt = int(math.log(n, prime))
        padic = sum(n // (prime ** np.arange(1, stopInt+1, dtype=int)))
        powers.append(padic)
        i += 1
    # All primes that are more than
    # half are guaranteed to only have
    # one power. So extend the powers
    # array by however many 1s...
    powers.extend([1] * (len(primes) - i))
    return np.array(primes, dtype=object), np.array(powers, dtype=object)
```
With this, we get that $K=34$, so $34!$ is an upper bound that has at least 8 million factors.
## Recursive function
We keep track of which prime we've chosen, the list of all prime factors, the current prime divisors that are part of our product, and the remaining limit.
* If our location goes off the end of the list, then we yield the number through the current prime divisors.
* Otherwise, we loop all possible exponents of the current prime, and recurse with each chosen power, decreasing the limit accordingly. For example, with our original limit of $34!$, say we had $2^2\times 3^2$ as our current running total. The exponent limit for 5 would be $\lceil \log_5\left(\frac{34!}{2^2\times3^2}\right)\rceil + 1$. In this way, our limit actually decreases fairly quickly, and the recursion depth isn't as deep as it would be.
* Exponents are only even, so we increment by 2.

```python
# file: "problem110.py"
def loopThroughNums(currDivLoc, currDivPowers, divisors, limit):
    # Base case is when we're over
    # the current divisor location.
    # In this case, we can yield a value.
    if currDivLoc >= len(currDivPowers):
        yield currDivPowers, np.prod(divisors ** currDivPowers)

    # For each possible divisor value
    # in the current location, yield all
    # possible values with that power...
    else:
        if currDivLoc == 0:
            loopLimit = math.ceil(math.log(limit, divisors[currDivLoc]))
        else:
            loopLimit = min(currDivPowers[currDivLoc - 1] + 1, math.ceil(math.log(limit, divisors[currDivLoc])))
        powerCopy = np.array(currDivPowers)
        # Only doing square numbers, so go by 2s
        for powerVal in range(0, loopLimit, 2):
            powerCopy[currDivLoc] = powerVal
            for product in loopThroughNums(currDivLoc + 1, powerCopy, divisors,
                                           limit / np.prod(divisors[currDivLoc] ** powerCopy[currDivLoc])):
                yield product
```
## Putting it all together
We have two functions, so combine them. At the end we take the minimum of our list.
```python
# file: "problem110.py"
solsRequired = 4000000
factLimit = (solsRequired + 1) * 2 - 1
# Get the first n primes where n is ceil(log_2(factLimit))
primes = np.array(primesieve.n_primes(math.ceil(math.log(factLimit, 2))), dtype=object)
# Get the factorial amount which has at least factLimit factors...
n = 2
_, powers = primeFactorFactorial(2)
while np.prod(powers + 1) < factLimit:
    n += 1
    _, powers = primeFactorFactorial(n)

print('{}! has at least {} solutions.'.format(n, solsRequired))

# Now we loop through all SQUARE numbers
# meaning prime factorizations with even numbers
# that are less than n! Any number with at least
# factLimit factors, we yield and collect the
# integer value to a list. Finally,
# we take the minimum of that list, which becomes
# our answer.
possibilities = []
count = 0
for powers, value in loopThroughNums(0, np.zeros(len(primes), dtype=object), primes, math.factorial(n)):
    count += 1
    if np.prod(powers + 1) > factLimit:
        possibilities.append(int(value ** 0.5))

print('Checked', count, 'values.')
print('Minimum value is', min(possibilities))
```
Running this code results in an output of,
```
34! has at least 4000000 solutions.
Checked 47204 values.
Minimum value is 93501300498606000600
8.538059700000002 seconds.
```
We see that there were 47204 values below 34! which had at least 8 million factors. Our answer is the minimum of these, which is **9350130049860600**.