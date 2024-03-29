---
layout: post
title: "#57 - Square root convergents"
date: 2017-06-19 20:11
number: 57
tags: [05_diff]
---
> Is it possible to show that the square root of two can be expressed as an infinite continued fraction.
> 
> $$
> \sqrt{2} = 1 + \frac{1}{2+\frac{1}{2 + \frac{1}{2+\dots}}} = 1.414213\dots
> $$
> 
> By expanding this for the first four iterations, we get:
> 
> $$
> \begin{aligned}
> 1 + \frac{1}{2} &= \frac{3}{2} = 1.5
> \\
> 1 + \frac{1}{2+\frac{1}{2}} &= \frac{7}{5} = 1.4
> \\
> 1 + \frac{1}{2+\frac{1}{2+\frac{1}{2}}} &= \frac{17}{12} = 1.41666\dots
> \\
> 1 + \frac{1}{2+\frac{1}{2+\frac{1}{2+\frac{1}{2}}}} &= \frac{41}{29} = 1.41379\dots
> \end{aligned}
> $$
> 
> The next three expansions are, $\frac{99}{70},\frac{239}{169},\frac{577}{408}$, but the eight expansion, $\frac{1393}{895}$, is the first example where the number of digits in the numerator exceeds the number of digits in the denominator.
> 
> In the first one-thousand expansions, how many fractions contain a numerator with more digits than denominator?
{:.lead}
* * *

We need to figure out a way to get from one expansion to the next. Recalculating every single expansion each time is inefficient.

Let's focus on the fraction part, as adding 1 is relatively easy. We can distill the question into: if the current expansion is $\frac{p}{q}$, then what is the next expansion?

From our problem example, the fractional-only expansion of the 2nd and 3rd step is $\frac{2}{5}$ and $\frac{5}{12}$ respectively. However, with the 3rd expansion, notice the expression in red:

$$
\frac{1}{2 + \color{red}{\frac{1}{2 + \frac{1}{2}}}}
$$
That is exactly the expression used for the 2nd expansion of $\frac{2}{5}$! In that case, we can plug this in directly, and end up computing 

$$
\frac{1}{2+\frac{2}{5}} = \frac{1}{\frac{12}{5}}=\frac{5}{12}
$$
In general, this pattern will follow each pair of expansion. As an extra example, the 3rd expansion can be plugged into the 4th in the same way: 

$$
\frac{1}{2 + \frac{5}{12}} = \frac{1}{\frac{29}{12}} = \frac{12}{29}
$$
With $\frac{p}{q}$, the next expansion becomes 

$$
\frac{1}{2+\frac{p}{q}} = \frac{1}{\frac{2q+p}{q}} = \frac{q}{2q+p}
$$ 


This calculation can easily be done in code quickly, as we only need the previous expansion. Adding 1 equates to adding the numerator and denominator together, so that is our check.
```python
# file: "problem057.py"
exceeding = 0
# Expansion of sqrt(2) is all 2s
num = 0
denom = 1
for _ in range(1000):
    num += 2 * denom
    # Flip
    temp = num
    num = denom
    denom = temp
    # Check if the length f num exceeds denom
    # ADD ONE!
    if len(str(num + denom)) > len(str(denom)):
        exceeding += 1
print(exceeding)
```
Running the above results in,
```
153
0.004323158785912579 seconds.
```
Therefore, there are **153** expansions within the first one-thousand that have more digits in the numerator than that of the denominator.