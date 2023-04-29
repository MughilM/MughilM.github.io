---
layout: post
title: "#35 - Circular primes"
date: 2016-05-10 11:34
number: 35
tags: [05_diff]
---
> The number, 197, is called a circular prime because all rotations of the digits: 197, 971, 719, are themselves prime.
> 
> There are thirteen such primes below 100: 2, 3, 5, 7, 11, 13, 17, 31, 37, 71, 73, 79, and 97.
> 
> How many circular primes are there below one million?
{:.lead}
* * *

We can convert the integer to a list of digits, and shift the rest of the digits to the left. In Python, this process is extremely simple. Below is the function:
```python
# file: "problem035.py"
def circulate(x):
    x = list(str(x))
    numOfCircles = len(x) - 1
    pList = []
    for _ in range(numOfCircles):
        # Copy the first digit.
        firstDigit = x[0]
        # Set all digits except
        # the last to the remaining
        x[:-1] = x[1:]
        # Set the last digit to the first
        # digit saved earlier
        x[-1] = firstDigit
        # Add to list
        pList.append(int(''.join(x)))
    return pList
```
This function is all good, but we still need a way to reduce the number of primes we need to check. There are two things we can observe:
* If we find a set of circular primes, each prime's final digit is each digit of the original prime. Therefore, a candidate circular prime **should not contain any even digits**. The only even circular prime is 2, we'll only be checking from 100 onwards.
* We can also remove the entire set of primes and not doubly check them later.

```python
# file: "problem035.py"
def containsAllOdd(x):
    return all([int(d) % 2 for d in str(x)])

limit = 1000000
# Generate primes between 100 and 1000000
primes = primesieve.primes(100, limit)
# Filter out all primes that have even digits.
primes = [p for p in primes if containsAllOdd(p)]
count = 13  # 13 primes that follow the property below 100.
for prime in primes:
    # Circulate, and see if each one
    # is in the list. If they are, then
    # save all of them and then
    # delete them, as we don't need to check them again.
    circlePrimes = circulate(prime)
    if all([p in primes for p in circlePrimes]):
        count += len(circlePrimes) + 1 # Plus 1 for current.
        # Delete.
        primes.remove(prime)
        for p in circlePrimes:
            primes.remove(p)
```
The output after running is,
```
55
0.8581394861900083 seconds.
```
Thus, there are **55** circular primes below one million. See below.

* 2
* 3
* 5
* 7
* 11
* 13, 31
* 17, 71
* 37, 73
* 79, 97
* 113, 131, 311
* 197, 971, 719
* 337, 373, 733
* 919, 199, 991
* 1193, 1931, 9311, 3119
* 3779, 7793, 7937, 9377
* 11939, 19391, 93911, 39119, 91193
* 19937, 99371, 93719, 37199, 71993
* 193939, 939391, 393919, 939193, 391939, 919393
* 199933, 999331, 993319, 933199, 331999, 319993