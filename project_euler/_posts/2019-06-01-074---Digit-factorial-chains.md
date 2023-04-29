---
layout: post
title: "#74 - Digit factorial chains"
date: 2019-06-01 17:31
number: 74
tags: [15_diff]
---
> The number 145 is well known for the property that the sum of the factorial of its digits is equal to 145:
>
> $$
> 1! + 4! + 5! = 1 + 24 + 120 = 145
> $$
>
> Perhaps less well known is 169, in that it produces the longest chain of numbers that link back to 169; it turns out that there are only three such loops that exist:
>
> $$
> \begin{aligned}
> 	& 169 \rightarrow 363601 \rightarrow 1454 \rightarrow 169
> 	\\
> 	& 871 \rightarrow 45361 \rightarrow 871
> 	\\
> 	& 872 \rightarrow 45362 \rightarrow 872
> \end{aligned}
> $$
>
> It is not difficult to prove that EVERY starting number will eventually get stuck in a loop. For example,
>
> $$
> \begin{aligned}
> 	& 69 \rightarrow 363600 \rightarrow 1454 \rightarrow 169 \rightarrow 363601\,(\rightarrow 1454)
> 	\\
> 	& 78 \rightarrow 45360 \rightarrow 871 \rightarrow 45361\,(\rightarrow 871)
> 	\\
> 	& 540 \rightarrow 145\,(\rightarrow 145)
> \end{aligned}
> $$
>
> Starting with 69 produces a chain of five non-repeating terms, but the longest non-repeating chain with a starting number below one million is sixty terms.
>
> How many chains, with a starting number below one million, contain exactly sixty non-repeating terms?
{:.lead}
* * *

To calculate the digit factorial sum, I'll be taking a mathematical approach, instead of converting to string and iterating, as that could slow things down. The ones digit in a number $n$ is obtained through $n\mod 10$. Meanwhile, $\lfloor n/10 \rfloor$ will delete the last digit. Using these two operations we can go through each digit without converting to a string.

Because we are dealing with chains, during calculation of the current chain, anytime we encounter a number whose chain length has already been calculated, we can immediately stop calculation and add the length. This drastically reduces the number of computations we need to do.

To account for the possibility that starting numbers under one million can have numbers greater than that in their chain, I'll use `defaultdict` to store the chain lengths.

We can also store the initial loop chain lengths to prevent any other numbers from falling into a loop. **40585 is actually another number that equals itself when taking the sum of the factorials of its digits**.
```python
# file: "problem074.py"
chainLengths = defaultdict(int)
# 1! = 1, 2! = 2, 145! ==> 145
# 40585! ==> 40585
# set these to 1
chainLengths[1] = 1
chainLengths[2] = 1
chainLengths[145] = 1
chainLengths[40585] = 1
# 871, 45361, 872, 45362
# have lengths of 2
for num in [871, 872, 45361, 45362]:
    chainLengths[num] = 2
# 169, 363601, 1454 have length 3
for num in [169, 363601, 1454]:
    chainLengths[num] = 3

# Calculate length
# of chains for everything less
# than 1 million
limit = 1000000
for n in range(3, limit):
    # See if it's been calculated
    if chainLengths[n] > 0:
        continue
    # Calculate until we've reached a
    # number that has been calculated...
    num = n
    length = 1
    visited = [num]
    while chainLengths[num] == 0:
        num = factorialDigitSum(num)
        visited.append(num)
        length += 1
    # Now set the visited numbers...
    # l is an offset.
    for l, k in enumerate(visited[::-1]):
        chainLengths[k] = l + chainLengths[num]

# Count how many numbers have chain length 60
countOfLength60 = 0
for num, length in chainLengths.items():
    if num < limit and length == 60:
        countOfLength60 += 1

print(countOfLength60)
```
A 0 means the chain hasn't been calculated yet. Running the code above results in,
```
402
3.9272377 seconds.
```
Thus, there are **402** terms with exactly 60 non-repeating terms in their chain. If we check the length of `chainLengths`, we see that there are 1000208, meaning we ended up storing 208 numbers above our limit.