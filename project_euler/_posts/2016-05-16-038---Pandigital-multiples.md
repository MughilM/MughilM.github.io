---
layout: post
title: "#38 - Pandigital multiples"
date: 2016-05-16 17:56
number: 38
tags: [05_diff]
---
> Take the number 192 and multiply it by each of 1, 2, and 3:
>
> $$
> 192\times1=192 \\
> 192\times2=384 \\
> 192\times3=576
> $$
>
> By concatenating each product we get the 1 to 9 pandigital, 192384576. We will call 192384576 the concatenated product of 192 and (1,2,3).
> 
> The same can be achieved by starting with 9 and multiplying by 1, 2, 3, 4, and 5, giving the pandigital, 918273645, which is the concatenated product of 9 and (1,2,3,4,5).
> 
> What is the largest 1 to 9 pandigital 9-digit number that can be formed as the concatenated product of an integer with (1,2,...,$n$) where $n$ > 1?
{:.lead}
* * *

With some trial and error, we can do some optimizations. Firstly, notice that the problem already gave us a large pandigital, 918273645. Therefore, in order for our pandigital to be bigger, it **must start with a 9**.

How long is $n$? Suppose that our test number $x$ is a 2-digit number starting with 9. Multiply by 1 and the result is 2-digits long. Multiplying by 2, 3, 4, ..., 9, 10, and 11 will get us three digits. However, a 1-9 pandigital is exactly 9 digits long by definition. This means that we are unable to get a concatenated product that is exactly 9 digits long, because the concatenated product of $x$ and (1, 2, 3) will have 8 digits, while including the next number 4 will send us over, with 11 digits.

With the same analysis assuming $x$ is 3 digits, you can see that $x$ with (1, 2) has 7 digits, but $x$ with (1, 2, 3) has 11 digits, so $x$ can't have 3 digits either.

However, **it is possible that $\mathbf{x}$ is 4 digits**. The concatenated product of $x$ with (1, 2) will have excatly 9 digits, so it's a candidate.

Additionally, $x$ can't have more than 4 digits either. If $x$ is $k$ digits long, then $2x$ is at least $k+1$ digits long. Therefore, any $k>4$ will result in a concatenated product exceeding 9 digits.

So we need a 4 digit number that starts with a 9, which is small enough to brute force. To test if a number is pandigital, we call `set()`. Since we are looking for the largest, we loop down from 10000.
```python
for i in range(10000, 9000, -1):
    result = int(str(i) + str(2 * i))
    if set(str(result)) == set('123456789'):
        print(result)
        break
```
Running this short loop, we get
```
932718654
0.001404046054606883 seconds.
```
Therefore, the largest 1-9 pandigital concatenated product is **932718654**.