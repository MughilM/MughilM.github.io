---
layout: post
title: "#6 - Sum square difference"
date: 2015-07-23 19:23
number: 6
tags: [05_diff]
---
> The sum of the squares of the first ten natural numbers is,
>
> $$ 
1^2 + 2^2 + \cdots + 10^2 = 385 $$
>
> The square of the sum of the first ten natural numbers is,
> 
> $$ 
(1 + 2 + \cdots + 10)^2 = 55^2 = 3025 $$
> 
> Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 3025 - 385 = **2640**.
>
> Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.
{:.lead}
* * *

Just directly loop until 100. No special packages are necassary, and we can use list comprehension to make the code extra short.

```python
# file: "problem006.py"
n = 100  
sum_square = sum(x ** 2 for x in range(n + 1))  
square_sum = sum(range(n + 1)) ** 2  
  
print(square_sum - sum_square)
```
Running gives,
```
25164150
5.570007488131523e-05 seconds.
```
## Bonus
We can also do this analytically. The expression we are asked to solve is

$$
S = \left( \sum_{i=1}^{100} i \right)^2 - \sum_{i=1}^{100} i^2
$$

Recall that $\sum_{i=1}^n i = \frac{n(n+1)}{2}$ and $\sum{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$. Substituting and solving, we get

$$
\begin{aligned}
    S &= \left( \frac{100(100+1)}{2} \right)^2 - \frac{100(100 + 1)(2(100) + 1)}{6}
    \\ &=
    5050^2 - 338350
    \\ &=
    \boxed{25164150}
\end{aligned}
$$