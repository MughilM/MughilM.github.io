---
layout: post
title: "#37 - Truncatable primes"
date: 2016-05-10 16:39
number: 37
tags: [05_diff]
---
> The number 3797 has an interesting property. Being prime itself, it is possible to continuously remove digits from left to right, and remain prime at each stage: 3797, 797, 97, and 7. Similarly we can work from right to left: 3797, 379, 37, 3.
> 
> Find the sum of the only eleven primes that are both truncatable from left to right and right to left.
> 
> 2, 3, 5, and 7 are not considered to be truncatable primes.
> {:.note}
{:.lead}
* * *

For terminology, I'll call a prime where you can chop off digits from left to right a **left truncatable prime**. Similarly, being able to chop off digits right to left will be called a **right truncatable prime**.

One way to do this is to generate primes which satisfy one of the two properties, and then check to see whether it satisfies the other property. Left truncatable primes are easier to generate, while right truncatable primes are simpler to check for validity.

When chopping off digits left to right, the resulting single digit **must be prime**. The single digit primes are 2, 3, 5, 7, but our prime can't end in 2, since that makes it even. Therefore, the prime must end in 3, 5, or 7

We can do a similar analysis for right-truncatable primes. Similar to left-truncatable, the left-most digit needs to be 2, 3, 5 or 7. Additionally, none of the inner digits can be even, since that would make a right-trunctabale prime composite at some point. 

To summarize our candidate prime generation:
* Start with 3, 5, and 7.
* Each time, add all odd numbers to the left of each number to make a new list. 
* Remove any numbers which aren't prime from the list. The new list only has numbers which are left truncatable.
* Test each prime to see if it's right-truncatable
* Repeat steps 2 through 4, each time adding digits to the left until all 11 primes are found.

For our prime list, I generate a sufficiently large set of primes that I use to check. Additionally, I've made a generator to give me the number each time it's been right truncated, so we can test these on the fly.
```python
# file: "problem037.py"
def rightTrunkNums(x):
    while x > 10:
        x //= 10
        yield x


primes = primesieve.primes(1000000)
# Keep adding digits to the right
# of a number which is left to right
# truncatable. We check these primes
# to see if they are left to right
# truncatable. Don't add 4, 6, 8
# because then it won't be truncatable
# the other direction
count = 0
# Our starting set are the one digit
# primes...
primeSet = [3, 5, 7]
# The numbers we can add to the left...
addingNumbers = [1, 2, 3, 5, 7, 9]
# While we haven't found all of them.
# We can deal with pure numbers, because
# each time, the length of the numbers
# goes up by one.
s = 0
tenPower = 1
while count < 11:
    # Add all of the possible digits to each
    # number in our list.
    primeSet = [[p + digit * (10 ** tenPower) for digit in addingNumbers]
                    for p in primeSet]
    # Flatten it.
    primeSet = [p for digitSet in primeSet for p in digitSet]
    # Go through, remove each number that isn't prime.
    primeSet = [p for p in primeSet if p in primes]
    # We are left with left truncatable primes by definition.
    # Now go through each prime and check if it's right
    # truncatable
    for p in primeSet:
        for num in rightTrunkNums(p):
            if num not in primes:
                break
        # Check if num is less than 10 and a prime...
        if num < 10 and num in primes:
            # We found one...
            print(p)
            count += 1
            s += p
        # else:
        #     primeSet.remove(p)
    # Increase power of ten...
    tenPower += 1

print()
print(s)
```
Running the code mass above, we get,
```
23
53
73
37
313
373
317
797
3137
3797
739397

748317
0.4748998125831603 seconds.
```
Therefore, the 11 primes we are looking for are 23, 53, 73, 37, 313, 373, 317, 797, 3137, 3797, and 739397. Their sum is **748317**.