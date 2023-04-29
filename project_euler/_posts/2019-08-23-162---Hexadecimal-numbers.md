---
layout: post
title: "#162 - Hexadecimal numbers"
date: 2019-08-23 15:58
number: 162
tags: [45_diff]
---
> In the hexadecimal number system numbers are represented using 16 different digits:
> 
> $$
> 0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F
> $$
> 
> The hexadecimal number AF when written in the decimal number system equals $10\times 16+15=175$.
>
> In the 3-digit hexadecimal numbers 10A, 1A0, A10, and A01 the digits 0, 1, and A are all present. Like numbers written in base ten we write hexadecimal numbers without leading zeroes.
>
> How many decimal numbers containing at most sixteen hexadecimal digits exist with all of the digits 0, 1, and A present at least once? Give your answer as a hexadecimal number.
>
> (A,B,C,D,E and F in upper case, without any leading or trailing code that marks the number as hexadecimal and without leading zeroes, e.g. 1A3F and not: 1a3f and not 0x1a3f and not $1A3F and not #1A3F and not 0000001A3F)
{:.lead}
* * *

In this case, we will count the complement as that is a smaller set. Then we will subtract it from the whole set of numbers. The complement is "hexadecimal numbers that are *missing* at least one of 0, 1, or A."

With $n$ digits (none starting with 0), we can choose any of the 15 non-zero numbers for the first digit, and then anythnig for the rest of the $n-1$ digits. Thus, in total there are $15\times 16^{n-1}$ $n$-digit hexadecimal numbers. From this, we **subtract** the number of hexadecimal numbers that are missing 0, 1, or A.

* Missing 0: With 0 removed, we have 15 choices for each digit, in total there are $15^n$.
* Missing 1: We have 15 choices, but only 14 for the first digit (0 is still there), so in total $14\times 15^{n-1}$.
* Missing A: Same as the above case, $14\times 15^{n-1}$.

So in total, there are $15^n+2\times 14\times 15^{n-1}$. However, we double-counted instances where the number is missing two of the above digits ("BA3" for example, is missing both 0 and 1). We **add** back in all numbers that have two of the above missing.
* Missing 0 and 1: There are 14 choices for each digit, so in total $14^n$.
* Missing 0 and A: Same as above case, $14^n$.
* Missing 1 and A: We have 13 choices for the first digit (0 is still there) and 14 for the rest, in total $13\times 14^{n-1}$.

The total we add back is then $2\times 14^n + 13\times 14^{n-1}$. However, we double-counted again, this time we added back in numbers that are missing all three. We need to subtract these off. The number of hexadecimal numbers that are missing all three is just $13^n$, as we have 13 choices for each digit.

In total, the number of $n$-digit numbers that have at least one 0, one 1, and one A is

$$
\begin{aligned}
	c(n) &= 
		\underbrace{15\times16^{n-1}}_\text{all numbers} -
		\underbrace{\left( 15^n+2\times14\times15^{n-1} \right)}_\text{missing one of 0, 1, A} + 
		\underbrace{\left( 2\times14^n+13\times14^{n-1} \right)}_\text{missing two of 0, 1, A} - 
		\underbrace{13^n}_\text{missing all 3}
	\\ &=
	15\times 16^{n-1} - 15^{n-1}(15+28) + 14^{n-1}(28+13)-13^n
	\\ &=
	\boxed{15\times 16^{n-1} - 43\times15^{n-1} + 41\times 14^{n-1} - 13^n}
\end{aligned}
$$

Since we have an expression for $c(n)$, all we have to do is sum the values up to $n=16$. The `hex()` function converts the final number to hexadecimal. It contains "0x" at the beginning so we index this out, and we can call `.upper()` to make all the letters uppercase.
```python
# file: "problem162.py"
countNums = lambda n: 15 * 16 ** (n-1) - 43 * 15 ** (n-1) + 41 * 14 ** (n-1) - 13 ** n
print(hex(sum(countNums(i) for i in range(3, 17)))[2:].upper())
```
Running this very short program, we have
```
3D58725572C62302
4.590000000004313e-05 seconds.
```
Therefore, our answer for the total hexadecimal numbers up to 16 digits is **3D58725572C62302**.
## Bonus
We can actually find a closed-form solution for the sum. We will use the fact that $\sum_{n=1}^N a^n = \frac{a^{N+1}-1}{a-1}-1$. Because $c(1)=c(2)=0$, we can start our summation from $n=1$. Therefore,

$$
\begin{aligned}
\sum_{n=1}^N c(n) &= 15\sum_{n=1}^N 16^{n-1} -
	43\sum_{n=1}^N 15^{n-1} + 
	41\sum_{n=1}^N 14^{n-1} -
	\sum_{n=1}^N 13^n
\\ &=
	15\left( \frac{16^N-1}{16-1}-1 \right) -
	43\left( \frac{15^N-1}{15-1}-1 \right) +
	41\left( \frac{14^N-1}{14-1}-1 \right) -
	\left( \frac{13^{N+1}-1}{13-1}-1 \right)
\\ &=
	16^N-1-15 - 
	\frac{43}{14}\left(15^N-1\right)+43 +
	\frac{41}{13}\left(14^N-1\right)-41 -
	\frac{1}{12}\left(13^{N+1}-1\right)+1
\\ &=
	16^N - \frac{43}{14}15^N + \frac{41}{13}14^N - \frac{1}{12}13^{N+1} - 13 + \frac{1}{1092}
\end{aligned}
$$

Plugging in $N=16$ and converting to hexadecimal gets us the same answer. The fractions will neatly cancel out, as 1092 has 14, 13, and 12 as factors.