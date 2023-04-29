---
layout: post
title: "#14 - Longest Collatz sequence"
date: 2015-07-27 11:07
number: 14
tags: [05_diff]
---
> The following iterative sequence is defined for the set of positive integers:
> 
> $$
> n = \begin{cases} \frac{n}{2},\quad &n\text{ is even} \\ 3n + 1, \quad &n\text{ is odd} \end{cases}
> $$
> 
> Using the rule above and starting with 13, we generate the following sequence
> 
> $$
> 13 \rightarrow 40 \rightarrow 20 \rightarrow 10 \rightarrow 5 \rightarrow 16 \rightarrow 8 \rightarrow 4 \rightarrow 2 \rightarrow 1
> $$
> 
> It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
> 
> Which starting number, under one million, produces the longest chain?
> 
> Once the chain starts, the terms are allowed to go over one million.
> {:.note}
{:.lead}
* * *

It is unfeasible to brute force every starting number, especially since we don't know how long the longest chain is. Instead, we can use dynamic programming. For example, when starting with 26, the next number is 13. However, we know that the sequence starting with 13 has 10 terms, so that means the sequence with 26 must have **11** terms. We can avoid a lot of repeated calculations using this method.

In code, we utilize a dictionary to save the sequence lengths of numbers we have already found. We can use `defaultdict` from the `collections` package to automatically initilazie all elements not explicitly set to 0.
```python
# file: "problem014.py"
limit = 1000000
seq_lengths = defaultdict(int)
seq_lengths[1] = 1

for n in range(2, limit + 1):
    sequence = [n]
    while seq_lengths[sequence[-1]] == 0:
        if sequence[-1] % 2 == 0:
            sequence.append(sequence[-1] // 2)
        else:
            sequence.append(3 * sequence[-1] + 1)
    # Set all numbers in the sequence to the right number.
    # We start at the first number that we have already,
    # and add the index in reverse.
    start_length = seq_lengths[sequence[-1]]
    for i, num in enumerate(sequence[::-1][1:], 1):
        seq_lengths[num] = start_length + i
# Return the max, filtering out starting numbers greater than the limit
max_n, max_len = max(((k, v) for k, v in seq_lengths.items() if k <= limit), key=lambda x: x[1])
print(f'{max_n} has the longest sequence starting under one million with {max_len} terms.')
```
Running our loop, our answer is thus,
```
837799 has the longest sequence starting under one million with 525 terms.
2.28711679999833 seconds.
```
