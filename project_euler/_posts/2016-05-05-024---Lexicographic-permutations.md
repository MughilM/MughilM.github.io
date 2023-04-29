---
layout: post
title: "#24 - Lexicographic permutations"
date: 2016-05-05 11:41
number: 24
tags: [05_diff]
---
> A permutation is an ordered arrangement of objects. For example, 3124 is one possible permutation of the digits 1, 2, 3, and 4. If all the permutations are listed numerically or alphabetically, we call it lexicographic order. The lexicographic permutations of 0, 1 and 2 are:
> 
> <p align="center">
>     012	021	102	120	201	210
> </p>
> 
> What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?
{:.lead}
* * *

There are $10! = 3\,628\,800$ permutations, which makes it infeasible to comb through them all. We need a way to find the next permutation that comes lexicographically after the current one. Fortunately, this is a common problem, and many have written about it. I will be using the algorithm used [here](https://www.nayuki.io/page/next-lexicographical-permutation-algorithm), which also provides an example.

The algorithm is as follows:
* Find the first element $x$ _from the right_ that does not follow non-increasing order.
* Swap $x$ with the _smallest_ element greater than $x$ in the non-increasing part of the string.
* Reverse the non-increasing part of the string.

Using Python's generators, we can generate the permutations on the fly and not waste any space. I also use a small helper function for step 1 in the process.
```python
# file: "problem024.py"
# Returns index of first element
# from RIGHT which isn't in
# non-increasing order.
def indexOfNonIncreasing(list_):
    j = len(list_) - 1
    while j > 0 and list_[j] <= list_[j-1]:
        j -= 1
    # If i is 0, then the whole list is
    # in non-increasing order.
    return j - 1

def genPermutations(charaSet):
    # While we haven't gotten to the
    # last permutation
    index = indexOfNonIncreasing(charaSet)
    while index != -1:
        # Find the least element greater
        # than what's at index...
        leastEle = index + 1
        while leastEle < len(charaSet) and charaSet[leastEle] > charaSet[index]:
            leastEle += 1
        # At this point we'll be one over, so subtract one.
        leastEle -= 1
        # Swap the numbers at the two locations
        temp = charaSet[index]
        charaSet[index] = charaSet[leastEle]
        charaSet[leastEle] = temp
        # Reverse the portion of non-increasingness
        charaSet[index + 1:] = charaSet[index + 1:][::-1]
        # Yield it.
        yield charaSet
        # Find the next index
        index = indexOfNonIncreasing(charaSet)
```
Now we run a loop until we hit the millionth one. One quirk with generators is that it runs once before it technically enters the for loop, so our iterator variable will actually be at 999999.
```python
# file: "problem024.py"
digits = list(range(10))
for i, perm in enumerate(genPermutations(digits), 1):
    # The generator runs once before it enters the loop,
    # so we need the permutation at 999999
    if i == 999999:
        print(''.join(map(str, perm)))
        break
```
After running, the output is
```
2783915460
0.9399120999732986 seconds.
```
Thus, **2783915460** is the millionth permutation.
## Bonus
It's actually possible to solve this problem using a basic calculator to find each digit from left to right. 

Notice that with the 10 digits we have on hand, there are exactly $9! = 362\,880$ permutations that start with each digit. Additionally, since we know that the permutations need to be in lexicographic order, the first $362\,880$ permutations will start with $0$, while the next $362\,880$ will start with $1$, and so on. Because $3\times362\,880>1\,000\,000$, the one millionth iteration starts with $2$. Additionally, the first permutation that starts with $2$ is $2013456789$, the $725\,761^{\text{th}}$ one. 

We can do the same analysis for the second digit. There are $1\,000\,000-725\,760 = 274\,240$ more permutations left. This time, there are $8! = 40\,320$ permutations that start with each of the digits, except for $2$, since we already used that. $\lceil274\,240 / 40\,320\rceil = 7$, so the seventh digit in our list, or $7$ will be the second digit in the one millionth digit.

So, in each step, we divide the remaining permutations to go by the number of ways to arrange the remaining digits, and take the ceiling to find which digit to use. In code, we do integer division i.e. the floor function, because we are dealing with 0-indexing. In the same manner, we decrement the number of permutations left by 1 before we divide. See below:

```python
# file: "problem024.py"
digits = list('0123456789')
permNum = 1000000
perm = ''
for i in range(1, 11):
    ways = math.factorial(10 - i)
    # It's permNum - 1 because integer
    # division is 0-index like.
    # If ways is 2, then the 4th permutation should
    # still be the 2nd digit, not the 3rd digit.
    loc = (permNum - 1) // ways
    digit = digits[loc]
    # Remove the element at loc.
    del digits[loc]
    perm += digit
    # Subtract
    permNum -= loc * ways

print(perm)
```

Running the above results in an output of,

```
2783915460
7.94073132950361e-05 seconds.
```

Notice the time is much faster, almost instantaneously.
