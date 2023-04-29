---
layout: post
title: "#65 - Convergents of $e$"
date: 2017-06-19 11:11
number: 65
tags: [15_diff]
---
> The square root of 2 can be written as an infinite continued fraction.
> 
> $$
> \sqrt{2} = 1 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2+\dots}}}}
> $$
> 
> The infinite continued fraction can be written, $\sqrt{2}=[1;(2)]$, $(2)$ indicates that 2 repeats *ad infinitum*. In a similar way, $\sqrt{23}=[4;(1,3,1,8)]$.
> 
> It turns out that the sequence of partial values of continued fractions for square roots provide the best rational approximations. Let us consider the convergents for $\sqrt{2}$.
> 
> $$
> \begin{aligned}
> 1 + \frac{1}{2} &= \frac{3}{2}
> \\
> 1 + \frac{1}{2 + \frac{1}{2}} &= \frac{7}{5}
> \\
> 1 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2}}} &= \frac{17}{12}
> \\
> 1 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2}}}} &= \frac{41}{29}
> \end{aligned}
> $$
> 
> Hence the sequence of the first ten convergents for $\sqrt{2}$ are:
> 
> $$
> 1,\frac{3}{2}, \frac{7}{5}, \frac{17}{12}, \frac{41}{29}, \frac{99}{70}, \frac{239}{169}, \frac{577}{408}, \frac{1393}{985}, \frac{3363}{2378}, \dots
> $$
> 
> What is most surprising is that the important mathematical constant,
>
> $$
> e=[2;1,2,1,1,4,1,1,6,1,\dots,1,2k,1,\dots]
> $$
> 
> The first ten terms in the sequence of convergents for $e$ are:
> 
> $$
> 2,3,\frac{8}{3},\frac{11}{4},\frac{19}{7}, \frac{87}{32}, \frac{106}{39}, \frac{193}{71}, \frac{1264}{465}, \frac{1457}{536},\dots
> $$
> 
> The sum of digits in the numerator of the $\text{10}^\text{th}$ convergent is $1+4+5+7=17$.
> 
> Find the sum of digits in the numerator of the $100^\text{th}$ convergent of the continued fraction for $e$.
{:.lead}
* * *

Here we are just calculating a fraction. We'll need to work from the bottom up. An example for the fourth convergent of $e$:

$$
2+\frac{1}{1 + \frac{1}{2+\frac{1}{1}}} = 2 + \frac{1}{1+\frac{1}{\color{red}{\frac{3}{1}}}}
= 2 + \frac{1}{1 + \frac{1}{3}} = 2 + \frac{1}{\color{red}{\frac{4}{3}}} = 2 + \frac{3}{4} =
\boxed{\frac{11}{4}}
$$
This allows us to see what the algorithm is. We start from the lowest fraction, and work our way up:

$$
\frac{1}{a_i + \frac{N}{D}} = \frac{1}{\frac{Da_i+N}{D}} = \frac{D}{Da_i+N}
$$
where $a_i$ is the current constant, $N$ is the current numerator and $D$ is the current denominator. This provides us with an easy loop for each constant in the continued fraction sequence.

The first step is generating all of our $a_i$'s. Since we need the 100th convergent, we generate 99 numbers (2 is considered the first iteration, without the fraction). We also need to keep track of the $2k$ constants as well. 
```python
# file: "problem065.py"
coeffs = []
n = 1
while len(coeffs) < 99:
    if len(coeffs) % 3 == 1:
        coeffs.append(n * 2)
        n += 1
    else:
        coeffs.append(1)
# Add and flip over and over again
num = 1
denom = coeffs[-1]
for i in range(len(coeffs) - 2, -1, -1):
    # Add
    num += coeffs[i] * denom
    # Flip
    temp = num
    num = denom
    denom = temp
# Add the final two
num += 2 * denom
print(sum([int(x) for x in str(num)]))
```
Running the code results in an output of,
```
272
0.00021688880320442342 seconds.
```
Therefore, the sum of the digits in the numerator of the 100th convergent of $e$ is **272**.

