---
layout: post
title: "#140 - Modified Fibonacci golden nuggets"
date: 2019-06-23 16:06
number: 140
tags: [55_diff]
---
> Consider the infinite polynomial series $A_G(x) = xG_1+x^2G_2+x^3G_3+\cdots$, where $G_k$ is the $k$th term of the second order recurrence relation $G_k=G_{k-1}+G_{k-2}$, $G_1=1$ and $G_2=4$; that is, $1,4,5,9,14,23, \dots$.
>
> For this problem we shall be concerned with values of $x$ for which $A_G(x)$ is a positive integer.
>
> The corresponding values of $x$ for the first five natural numbers are shown below.
>
> | $x$                       | $A_G(x)$ |
> | --------------------------- | ---------- |
> | $\frac{\sqrt{5}-1}{4}$    | 1          |
> | $\frac{2}{5}$             | 2          |
> | $\frac{\sqrt{22}-2}{6}$   | 3          |
> | $\frac{\sqrt{137}-5}{14}$ | 4          |
> | $\frac{1}{2}$             | 5          |
>
> We shall call $A_G(x)$ a golden nugget if $x$ is rational, because they become increasingly rarer; for example, the 20th golden nugget is 211345365.
>
> Find the sum of the first thirty golden nuggets.
{:.lead}
* * *

This problem is very similar to [#137 - Fibonacci golden nuggets](/blog/project_euler/2019-06-21-137-Fibonacci-golden-nuggets){:.heading.flip-title} where we dealt with Fibonacci numbers. These are Fibonacci-like, but not quite the same sequence. However, it's natural to assume a similar pattern will follow, and so we will do the same steps. First, we'll rewrite the sequence definition as a closed form formula:

$$
\begin{aligned}
	A_G(x) &= \sum_{n=1}^\infty G_n x^n
	\\ &=
	G_1x + G_2x^2 + \sum_{n=3}^\infty G_n x^n
	\\ &=
	x + 4x^2 + \sum_{n=3}^\infty \left(G_{n-1} + G_{n-2}\right)x^n
	\\ &=
	x + 4x^2 + x\sum_{n=3}^\infty G_{n-1}x^{n-1} + x^2\sum_{n=3}^\infty G_{n-2}x^{n-2}
	\\ &=
	x + 4x^2 + x\sum_{n=2}^\infty G_n x^n + x^2\sum_{n=1}^\infty G_n x^n
	\\ &=
	x + 4x^2 + x\left(A_G(x) - x\right) + x^2A_G(x)
	\\ &=
	x + 3x^2 + xA_G(x) + x^2A_G(x)
	\\
	A_G(x) &= \frac{x+3x^2}{1-x-x^2}
\end{aligned}
$$

Let $A_G(x) = C$ and use the quadratic formula to solve for $x$:

$$
\begin{aligned}
	C &= \frac{x+3x^2}{1-x-x^2}
	\\
	x^2(C+3)+x(C+1)-C &= 0
	\\
	x &= \frac{-(C+1)\pm \sqrt{(C+1)^2-4(C+3)(-C)}}{2(C+3)}
	\\ &=
	\frac{-(C+1)\pm\sqrt{(C+1)^2+4C^2+12C}}{2(C+3)}
	\\ &=
	\frac{-(C+1)\pm\sqrt{5C^2+14C+1}}{2(C+3)}
\end{aligned}
$$

The discriminant needs to be a perfect square to yield an integer. Let's first find a few values of $C$ that results in a perfect square and see if we recognize a pattern.
| $C$     | Root of Discriminant | $x$             |
| ----- | -------------------- | ----------------- |
| 2     | 7                    | $\frac{2}{5}$   |
| 5     | 14                   | $\frac{1}{2}$   |
| 21    | 50                   | $\frac{7}{12}$  |
| 42    | 97                   | $\frac{3}{5}$   |
| 152   | 343                  | $\frac{19}{31}$ |
| 296   | 665                  | $\frac{8}{13}$  |
| 1050  | 2351                 | $\frac{50}{81}$ |
| 2037  | 4558                 | $\frac{21}{34}$ |
We see the Fibonacci fractions, but also other fractions. These non-Fibonacci fractions also follow their own Fibonacci rule (2, 5, 7, 12, ...). These fractions come from adding the Fibonacci to the $G_n$ sequence:

$$
\{1,1,2,3,5,8,\dots\}\,+\,\{1,4,5,9,14,23,\dots\} = \{2,5,7,12,19,31,\dots\}
$$

With this fact, we can conclude that the $x$-value associated with the $n$th golden nugget is

$$
x_n=\begin{cases}
	\frac{F_n+G_n}{F_{n+1}+G_{n+1}} \qquad &n\text{ is odd}
	\\
	\frac{F_n}{F_{n+1}} \qquad &n\text{ is even}
\end{cases}
$$

Let $H_n = F_n + G_n$. We can plug these into the function and get formulas for the $n$th golden nugget $C_n$:

$$
C_n=\begin{cases}
	\frac{H_nH_{n+1}+3H_n^2}{11} \qquad &n\text{ is odd}
	\\
	F_nF_{n+1}+3F_n^2 \qquad &n\text{ is even}
\end{cases}
$$

Like with the regular Fibonacci numbers, $H_n$ also has it's own "Cassini's Identity". In this case, $H_{n+1}H_{n-1}-H_n^2=11(-1)^{n+1}$. We can prove this by induction, just like the normal Cassini's Identity.

Now that we have a direct formula, it is simple to code a loop to calculate the sum of the first 30 golden nuggets.
```python
# file: "problem140.py"
H = [3, 2, 5]
F = [0, 1, 1]
for _ in range(3, 32):
    H.append(H[-2] + H[-1])
    F.append(F[-2] + F[-1])

# Now compute the sum of the first 30 thirty nuggets
goldenNuggetSum = 0
for n in range(1, 31):
    if n % 2 == 1:
        goldenNuggetSum += (H[n] * H[n + 1] + 3 * H[n] ** 2) // 11
    else:
        goldenNuggetSum += F[n] * F[n + 1] + 3 * F[n] ** 2

print(f'The sum of the first 30 golden nuggets is {goldenNuggetSum}.')
```

Running this short code, we get
```
The sum of the first 30 golden nuggets is 5673835352990.
5.6199999999950734e-05 seconds.
```
Therefore, our golden nugget sum is **5673835352990**.