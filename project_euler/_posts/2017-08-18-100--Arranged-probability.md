---
layout: post
title: "#100 - Arranged probability"
date: 2017-08-18 01:05
number: 100
tags: [30_diff]
---
> If a box contains twenty-one coloured discs, composed of fifteen blue discs and six red discs, and two discs were taken at random, it can be seen that the probability of taking two blue discs $P(BB)=(15/21)\times(14/20)=1/2$.
> 
> The next such arrangement, for which there is exactly 50% chance of taking two blue discs at random, is a box containing eighty-five blue discs and thirty-five red discs.
> 
> By finding the first arrangement to contain over $10^{12}=1\,000\,000\,000\,000$ discs in total, determine the number of blue discs that the box would contain.
{:.lead}
* * *

If we have $n$ total discs and $b$ blue discs in the box (we don't replace the first disc one we've taken it out), then the probability of taking 2 blue discs is

$$
P(BB)=\left(\frac{b}{n}\right)\left(\frac{b-1}{n-1}\right)
$$

We want this to be $1/2$, so let's set this equal and try to simplify:

$$
\begin{aligned}
	\left(\frac{b}{n}\right)\left(\frac{b-1}{n-1}\right) &=
		\frac{1}{2}
	\\
	\frac{b^2-b}{n^2-n} &= \frac{1}{2}
	\\
	2b^2-2b &= n^2-n
	\\
	2b^2-2b+n-n^2 &= 0
\end{aligned}
$$

We have a quadratic in $b$, so apply the quadratic formula and solve it:

$$
\begin{aligned}
	b &= \frac{2\pm\sqrt{4-4(2)(n-n^2)}}{4}
	\\ &=
	\frac{2\pm2\sqrt{1-2n+2n^2}}{4}
	\\ &=
	\frac{1}{2}\left(1 + \sqrt{2n^2-2n+1}\right)
    \\ &=
    \frac{1}{2}\left(1 + \sqrt{n^2+(n-1)^2}\right)
\end{aligned}
$$

So this means the number of blue discs will only be an integer if $n^2+(n-1)^2$ is a perfect square. This is a sum of two squares which needs to equal another square, so another way to frame the restriction is that if $z=\sqrt{n^2+(n-1)^2}$, then $\{n, n-1, z\}$ has to be a Pythagorean triple.

Looking at the [Tree of Pythagorean triples](https://en.wikipedia.org/wiki/Tree_of_primitive_Pythagorean_triples), repeatedly multiplying by the $B$ matrix will preserve the property that the two smaller sides are within one unit of each other.

$$
B=\begin{bmatrix}
	1 & 2 & 2 \\
	2 & 1 & 2 \\
	2 & 2 & 3
\end{bmatrix}
$$

Tha value of $n$ will be the larger of the two sides, which during the multiplication, will alternate being the first and second value ($\{3,4,5\}\rightarrow \{21,20,29\}\rightarrow \{119,120,169\}\rightarrow\dots$). 85 blue discs is associated with $n=120$.

Both the alternation and the matrix multiplication are simple to code. We calculate $n$, then $b$.
```python
# file: "problem100.py"
b = 15
n = 21
switch = 0
while n < 10 ** 12:
    n = 3 * n - 1 + 2 * (2 * b - 1)
    if switch:
        n -= 1
    b = (int((2 * n ** 2 - 2 * n + 1) ** 0.5) + 1) // 2

print('b:', b)
print('n:', n)
```
Running our code, we get
```
b: 756872327473
n: 1070379110497
0.00012419999999999792 seconds.
```
Thus, we have **756872327473** blue discs in the box when the number of total discs first surpasses $10^{12}$.