---
layout: post
title: "#113 - Non-bouncy numbers"
date: 2019-02-11 13:46
number: 113
tags: [30_diff]
---
> Working from left-to-right if no digit is exceeded by the digit to its left it is called an increasing number; for example, 134468.
>
> Similarly if no digit is exceeded by the digit to its right it is called a decreasing number; for example, 66420.
>
> We shall call a positive integer that is neither increasing nor decreasing a "bouncy" number; for example, 155349.
>
> As $n$ increases, the proportion of bouncy numbers below $n$ increases such that there are only 12951 numbers below one-million that are not bouncy and only 277032 non-bouncy numbers below $10^{10}$.
>
> How many numbers below a googol ($10^{100}$) are not bouncy?
{:.lead}
* * *

With [#112 - Bouncy numbers](/blog/project_euler/2019-02-11-112---Bouncy-numbers){:.heading.flip-title}, we explicitly tested whether each number was bouncy. However, in this case, the limit is massive, which means we must take a clever counting approach.

From the problem, a number will fall into 3 distinct groups: an increasing number, a decreasing number, or neither. The former two collectively refer to non-bouncy numbers. 

## Increasing numbers
One important note is that the number of increasing numbers differs depending on **what digit the number starts with.** For example, there are 5 two-digit increasing numbers that start with 5 (55-59), but only 2 that start with 8 (88 and 89). It appears, at least in the two-digit case, that the number of increasing numbers starting with digit $d$ is $10-d$.

In the 3-digit case, something interesting case. Let's say our 3-digit number starts with 4. This means the next digit can be anything between 4-9. However, now we are essentially finding all two-digit increasing numbers that start with 4-9. The sum of these gives us the number of 3-digit increasing numbers that start with 4. The same logic can be used to see how many there are that start with each digit.

### A quick example with 4-digit numbers
We keep track of the number of increasing numbers per starting digit, except 0 because numbers can't start with 0, and an increasing number can't contain a 0.

* Our array looks like [1, 1, 1, 1, 1, 1, 1, 1, 1], because there's only 1 number for each starting digit.
* For two digits, the array now looks like [9, 8, 7, 6, 5, 4, 3, 2, 1] i.e. there are 9 2-digit numbers that start with 1 (11-19), 8 that start with 2 (22-29), ...
* For three digits, the array looks like [45, 36, 28, 21, 15, 10, 6, 3, 1]. The first element is calculated by 9 + 8 + 7 + ... + 1 = 45, the second element is calculated by 8 + 7 + 6 + ...., because 3-digit increasing numbers that start with 2 can't have 1 as a second digit. This is a cumulative sum.
* For four digits, the array looks like [165, 120, 84, 56, 35, 20, 10, 4, 1]. It should make sense that the right-most number is 1, since there's always only 1 increasing number that starts with 9, and that is 99...99.

Using `np.cumsum` we can do this operation in one line.
## Decreasing numbers
For decreasing numbers, the method is exactly the same, just that the cumulative sum travels in the other direction. Additionally, we need to include 0 in our calculations. This is because it is possible for decreasing numbers to have zeroes in them i.e. 2100 is a decreasing number.

## Double-counting
If we add the number of increasing and decreasing numbers, we'll almost have the answer, but we have double counted numbers with **all the same digit**. A number like 5555 is both increasing and decreasing. We need to subtract these out (as well the "number" of all 0s which we have counted as a decreasing number). There are 10 of these in total (0s, 1s, 2s, ...).

## Code
`numpy` makes this consice.
```python
# file: "problem113.py"
incrNums = np.ones(9, object)
decrNums = np.ones(10, object)  # Count 0 as a decreasing digit
# All one digit #s are not bouncy
total = 9
digitMax = 100

for digit in range(2, digitMax+1):
    # Calculate number of increasing and decreasing numbers for the current digit length
    incrNums = np.cumsum(incrNums)
    decrNums = np.cumsum(decrNums)
    # Add the total of each of these to the total.
    # However, we have to subtract 10, because one we counted is 00...0, which isn't a number.
    # We also double-counted numbers that go like 1...1, 22...2, ..., 99...9; there are nine of them.
    total += np.sum(incrNums) + np.sum(decrNums) - 10

print('There are', total, 'non-bouncy numbers below 10 ^', digitMax)
```
Running the loop gets us,
```
There are 51161058134250 non-bouncy numbers below 10 ^ 100
0.002906500012613833 seconds.
```
Therefore, there are **51161058134250** non-bouncy numbers below a googol.