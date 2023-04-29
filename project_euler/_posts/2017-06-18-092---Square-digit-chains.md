---
layout: post
title: "#92 - Square digit chains"
date: 2017-06-18 10:18
number: 92
tags: [05_diff]
---
> A number chain is created by continuously adding the square of the digits in a number to form a new number until it has been seen before.
> 
> For example,
> 
> $$
> 44\rightarrow 32\rightarrow 13\rightarrow 
> 10\rightarrow\mathbf{1}\rightarrow\mathbf{1}
> \\
> 85\rightarrow\mathbf{89}\rightarrow145\rightarrow 42\rightarrow20
> \rightarrow4\rightarrow16\rightarrow37\rightarrow58\rightarrow\mathbf{89}
> $$
> 
> Therefore any chain that arrives at 1 or 89 will become stuck in an endless loop. What is most amazing is that EVERY starting number will eventually arrive at 1 or 89.
> 
> How many starting numbers below ten million will arrive at 89?
{:.lead}
* * *

One important observation is that **the ordering of the digits do not matter**. For example, the numbers 5332, 5323, 5233, 3532, 3352, 3325, 3235, 3253, 3523, 2335, 2353, and 2533 all compute to a digit squared sum of $25+9+9+4=\mathbf{47}$. We can take advantage of this fact to avoid visiting each number and calculating its chain.

Our search space will eventually consist of numbers of the form $abcd$, where $a\geq b\geq c\geq d$. The number 5332 from above would be part of this set, **but none of its permutations**, so this sum will only be calculated once.

But remember 5332 had **11 other permutations**, which **all end at 89** and need to be accounted for. So, for each number in our set, we have to calculate the **number of distinct permutations it has.** On its own, it is not too complicated to calculate, but we have to watch out for is repeated digits. 

If all our digits are different, then there are exactly $4! = 24$ ways of ordering them. However, there are only 12 permutations of 5332, because there are 2 3s. We have to divide by the number of ways these two 3s can be arranged, which is just 2, and so $24/2 = 12$. In general, if $n_d$ is the number of times the digit $x$ appears in the number, then **the number of permutations of our 4-digit number is**

$$
Perm(abcd) = \frac{24}{c_0!c_1!\dots c_9!} = 
	\frac{24}{\prod_{d=0}^9 c_d!}
$$
What about leading 0s? If our original number was 8400, then the above formula also counts "numbers" such as 0480 and 0084. But this actually works in our favor because we are counting numbers with less than 4 digits! So no external correction is needed here.
{:.note}

To generate the number set, I use `yield` and create a generator. This way, the entire list isn't stored in memory. The looping algorithm is a recursive one, which keeps track of the remaining length of the number. The base case is if the length is 0, in which case we return the number. Otherwise, we append each possible digit, recurse with each of them, and decrement the length by 1. I do not include 0 with the initial loop, since we can't have leading zeros.

The function is to calculate the sum of square digits is using integer division 10 to get each digit, as opposed to converting to a string, which saves some time. We also hav a separate function to count the number of permutations.
```python
# file: "problem092.py"
def sumOfSquareOfDigits(num):
    s = 0
    while num != 0:
        s += (num % 10) ** 2
        num = num // 10
    return s


def loopNumbers(length, currNum=None):
    if length == 0:
        yield currNum
        return
    # Top level
    if currNum is None:
        # Don't start with a 0
        for n in range(1, 10):
            for number in loopNumbers(length - 1, n):
                yield number
    # Any other level...
    else:
        for n in range(currNum % 10 + 1):  # <-- Zero allowed
            for number in loopNumbers(length - 1, currNum * 10 + n):
                yield number

def countPermutations(n):
    nstr = str(n)
    digits = '0123456789'
    return math.factorial(len(nstr)) // \
        np.prod([math.factorial(nstr.count(digit)) for digit in digits])


is89 = []
is1 = []
total89 = 0
for number in loopNumbers(7):
    count = countPermutations(number)
    i = number
    startingSquare = sumOfSquareOfDigits(i)
    # Check to see if it's in the 89 list...
    if startingSquare in is89:
        total89 += count
    elif startingSquare in is1:
        continue
    else:
        while i != 89 and i != 1:
            i = sumOfSquareOfDigits(i)
        if i == 89:
            is89.append(startingSquare)
            total89 += count
        else:
            is1.append(startingSquare)
print(total89)
```
Running our code gets an output of,
```
8581146
0.2859901000000001 seconds.
```
Thus, we have **8581146** numbers below 10 million that end in 89.
