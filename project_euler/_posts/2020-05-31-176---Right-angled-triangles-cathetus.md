---
layout: post
title: "#176 - Right-angled triangles that share a cathetus"
date: 2020-05-31 19:59
number: 176
tags: [70_diff]
---
> The four right-angled triangles with sides (9, 12, 15), (12, 16, 20), (5, 12, 13) and (12, 35, 37) all have one of the shorter sides (catheti) equal to 12. It can be shown that no other integer sided right-angled triangle exists with one of the catheti equal to 12.
> 
> Find the smallest integer that can be the length of a cathetus of exactly 47547 different integer sided right-angled triangles.
{:.lead}
* * *

On the Wolfram Mathworld page for [Pythagorean Triples](https://mathworld.wolfram.com/PythagoreanTriple.html), we find the following formula for the number of triangles with leg length $s$.

$$
L(s) = \begin{cases}
    \frac{1}{2}[(2a_1+1)(2a_2+1)\cdots(2a_n+1)-1] \qquad &\text{for }a_0=0
    \\
    \frac{1}{2}[(2a_0-1)(2a_1+1)(2a_2+1)\cdots(2a_n+1)-1] \qquad &\text{for }a_0\geq 1
\end{cases}
$$

where $s=2^{a_0}p_1^{a_1}\cdots p_n^{a_n}$.

We can verify this using the example given in the problem. $s=12=2^2\times 3$. Therefore, since $a_0=2\geq 1$, we use the second case, and $L(12) = \frac{1}{2}[(2(2)-1)(2(1)+1)-1] = \frac{1}{2}(3(3)-1) = 4$.

But we're not asked to find $L(s)$. Instead, we are given what $L(s)$ is, and asked to find the minimum value of $s$. We want $L^{-1}(s)$.

The first thing we do is adjust the formula to accommodate the inverse. Assuming $s$ is broken down into its prime factorization, we multiply by 2 and 1 to $L(s)$, and keep the product on the other side. Because we want the **smallest** $s$, we want $s$ to have powers of 2, and not potentially anything larger. (12 has a power 12, 15 does not, but both evaluate to 4 triangles).

$$
(2a_0-1)(2a_1+1)(2a_2+1)\cdots(2a_n+1)=2L(s)+1
$$

On the left, we are multiplying numbers together, so maybe if we find the prime factorization of $\mathbf{2L(s)+1}$, we can be a step closer to the answer.

The factorization of $2(47547)+1=95095$ is $5\times 7\times 11\times 13\times 19$. We have 5 factors, which can be split among 5 factors above (corresponding to $a_0,a_1,\dots,a_4$). But which factor corresponds to which $a_i$? Remember $a_i$ corresponds to the exponent in the original factorization of $s$. In order to achieve the smallest product possible, we want the **largest exponents on the smallest prime factors** (and vice versa). Therefore, we want

$$
\begin{cases}
    2a_0-1 &= 19
    \\
    2a_1+1 &= 13
    \\
    2a_2+1 &= 11
    \\
    2a_3+1 &= 7
    \\
    2a_4+1 &= 5
\end{cases}
$$

These are simple equations, and the exponents for $s$ are then $\{a_0,a_1,a_2,a_3,a_4\}=\{10,6,5,3,2\}$. Thus, the smallest $s$ which makes $L(s)=47547$ is

$$
s=2^{10}\times 3^6\times 5^5\times 7^3\times 11^2 = \boxed{96818198400000}
$$

Our large answer is **96818198400000**. No code necessary for this problem :). This worked because the factorization of $2L(s)+1$ had 5 clean factors we could evenly distribute. If the factorization were more complicated, we would have to check additional cases.