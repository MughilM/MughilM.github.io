---
layout: post
title: "#137 - Fibonacci golden nuggets"
date: 2019-06-21 14:19
number: 137
tags: [50_diff]
---
> Consider the infinite polynomial series $A_F(x) = xF_1 + x^2F_2 + x^3F_3 + \dots$, where $F_k$ is the $k^{\text{th}}$ term in the Fibonacci sequence: 1, 1, 2, 3, 5, 8, ...; that is, $F_k = F_{k-1} + F_{k-2}$, $F_1=1$ and $F_2=1$.
>
> For this problem we shall be interested in values of $x$ for which $A_F(x)$ is a positive integer.
> 
> Surprisingly,
> 
> $$
> \begin{aligned}
> 	A_F\left(\frac{1}{2}\right) &= \left(\frac{1}{2}\right)(1) + \left(\frac{1}{2}\right)^2(1) + \left(\frac{1}{2}\right)^3(2) + \left(\frac{1}{2}\right)^4(3) + \cdots
> 	\\ &=
> 	\frac{1}{2} + \frac{1}{4} + \frac{2}{8} + \frac{3}{16} + \frac{5}{32} + \cdots
> 	\\ &= 2
> \end{aligned}
> $$
> 
> The corresponding values of $x$ for the first five natural numbers are shown below.
> 
> | $\mathbf{x}$            | $\mathbf{A_F(x)}$ |
> | ------------------------- | ------------------- |
> | $\sqrt{2}-1$            | 1                   |
> | $\frac{1}{2}$           | 2                   |
> | $\frac{\sqrt{13}-2}{3}$ | 3                   |
> | $\frac{\sqrt{89}-5}{8}$ | 4                   |
> | $\frac{\sqrt{34}-3}{5}$ | 5                   |
> 
> We shall call $A_F(x)$ a golden nugget if $x$ is rational, because they become increasingly rarer; for example, the 10th golden nugget is 74049690.
> 
> Find the 15th golden nugget.
{:.lead}
* * *

I had to do a bit of research as to when exactly the golden nuggets occur, but it was worth it. Firstly though, infinite series are a tad annoying to work with, so I came up with a closed form formula for it.

$$
\begin{aligned}
	A_F(x) &= \sum_{n=1}^\infty F_nx^n
	\\ &=
	F_1x + F_2x^2 + \sum_{n=3}^\infty F_nx^n
	\\ &=
	x + x^2 + \sum_{n=3}^\infty(F_{n-1} + F_{n-2})x^n
	\\ &=
	x + x^2 + x\sum_{n=3}^\infty F_{n-1}x^{n-1} + x^2\sum_{n=3}^\infty F_{n-2}x^{n-2}
	\\ &=
	x + x^2 + x\sum_{n=2}^\infty F_nx^n + x^2\sum_{n=1}^\infty F_nx^n
	\\ &=
	x + x^2 + x(A_F(x) - x) + x^2A_F(x)
	\\ &=
	x + xA_F(x) + x^2A_F(x)
	\\
	A_F(x) &= \frac{x}{1-x-x^2}
\end{aligned}
$$

We must find when this equals an integer $C$. Since we have a quadratic equation $x$, we can solve directly,

$$
\begin{aligned}
	C &= \frac{x}{1-x-x^2}
	\\
	Cx^2 + x(C+1)-C &= 0
	\\
	x &= \frac{-(C+1)\pm\sqrt{(C+1)^2-4C(-C)}}{2C}
	\\ &=
	\frac{-(C+1)\pm\sqrt{(C+1)^2+(2C)^2}}{2C}
\end{aligned}
$$

Regardless of what happens with the other terms, the **discriminant has to evaluate to a perfect square.** This will lead to $x$ being rational and a golden nugget.

## Finding when the discriminant is a perfect square
This is where I had to some digging, and found a great paper by Dae S. Hong, [When is the Generating Function of the Fibonacci Numbers an Integer?](https://www.researchgate.net/profile/Dae-Hong-3/publication/280034292_When_Is_the_Generating_Function_of_the_Fibonacci_Numbers_an_Integer/links/627db982973bbb29cc79d231/When-Is-the-Generating-Function-of-the-Fibonacci-Numbers-an-Integer.pdf). All credit goes to the author for the insights. I've purposely written it as a sum of two squares to signify that we are in fact looking for Pythagorean triples of that form. I experimented and found the first 4 values where it results in a perfect square.
| $C$ | Root of Discriminant | $x$             |
| ----- | -------------------- | ----------------- |
| 2     | 5                    | $\frac{1}{2}$   |
| 15    | 34                   | $\frac{3}{5}$   |
| 104   | 233                  | $\frac{8}{13}$  |
| 714   | 1597                 | $\frac{21}{34}$ |
Notice the numerator and denominator of the $x$ values. They are consecutive Fibonacci numbers! This shouldn't come as a surprise since we started with a Fibonacci series.

If we assume $x=\frac{F_n}{F_{n+1}}$ and plug it into $A_F(x)$, we get

$$
\begin{aligned}
A_F\left(\frac{F_n}{F_{n+1}}\right) &= \frac{\frac{F_n}{F_{n+1}}}{1-\frac{F_n}{F_{n+1}}-\left(\frac{F_n}{F_{n+1}}\right)^2}
\\ &=
\frac{F_nF_{n+1}}{F_{n+1}^2-F_nF_{n+1}-F_n^2}
\\ &=
\frac{F_nF_{n+1}}{F_{n+1}(F_{n+1}-F_n)-F_n^2}
\\ &=
\frac{F_nF_{n+1}}{F_{n+1}F_{n-1}-F_n^2}
\end{aligned}
$$

[Cassini's Identity](https://en.wikipedia.org/wiki/Cassini_and_Catalan_identities) can result in simplifying all the way through,

$$
A_F\left(\frac{F_n}{F_{n+1}}\right) = (-1)^nF_nF_{n+1}
$$

We get an integer! With reference to the table above, $A_F(x)=2$ corresponds to $n=2$, and $A_F(x)=15$ corresponds to $n=4$. Our golden nuggets occur when $n$ is even. The reason is that when $n$ is odd, $(-1)^n$ will result in $A_F(x)$ being negative. However, $x$ is still positive, and when plugging that in to our infinite series, means the series is positive as well. Thus, having $n$ be odd is a mismatch to the infinite series definition.

## Evaluation
Since $n$ is even, we can write it as $2k$. This means the $k^{\text{th}}$ golden nugget is

$$
A_F\left(\frac{F_{2k}}{F_{2k+1}}\right) = (-1)^{2k}F_{2k}F_{2k+1} = F_{2k}F_{2k+1}
$$
We can verify from the problem statement that $k=10$ should result in 74049690 and we see it immediately:

$$
F_{2(10)}F_{2(10)+1}=F_{20}F_{21}=6765(10946)=74049690
$$

Thus, to get the answer to the problem, **just plug in 15!**

$$
F_{2(15)}F_{2(15)+1}=F_{30}F_{31}=832040(1346269)=1120149658760
$$
Therefore, the 15th golden nugget is **1120149658760**. No code required, unless we want to write a Fibonacci generator to find the 30th and 31st Fibonacci numbers.