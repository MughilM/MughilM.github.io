---
layout: post
title: "#108 - Diophantine reciprocals I"
date: 2019-08-20 17:36
number: 108
tags: [30_diff]
---
> In the following equation $x$, $y$, and $n$ are positive integers.
> 
> $$
> \frac{1}{x} + \frac{1}{y} = \frac{1}{n}
> $$
> 
> For $n=4$ there are exactly three distinct solutions:
> 
> $$
> \begin{aligned}
    \frac{1}{5} + \frac{1}{20} &= \frac{1}{4}
    \\
    \frac{1}{6} + \frac{1}{12} &= \frac{1}{4}
    \\
    \frac{1}{8} + \frac{1}{8} &= \frac{1}{4}
\end{aligned}
> $$
> 
> What is the least value of $n$ for which the number of distinct solutions exceeds one-thousand?
> 
> This problem is an easier verison of [#110 - Diophantine reciprocals II](/blog/project_euler/2019-08-21-110-Diophantine-reciprocals-II){:.heading.flip-title}; it is strongly advised that you solve this one first.
{:.lead}
* * *

Since we are dealing with positive fractions, both $x$ and $y$ should be strictly greater than $n$. Suppose $x$ and $y$ exceed $n$ by $a$ and $b$ respectively. Then, we can rewrite the equation as follows:

$$
\begin{aligned}
    \frac{1}{a+n} + \frac{1}{b+n} &= \frac{1}{n}
    \\
    \frac{a+b+2n}{(a+n)(b+n)} &= \frac{1}{n}
    \\
    ab + an + bn + n^2 &= an + bn + 2n^2
    \\
    ab &= n^2
\end{aligned}
$$

The conclusion is that any values $a$ and $b$ which multiply together to get $n^2$ are valid solutions to the original problem. For example, with $n=4$, we have $n^2=16$. There are 3 pairs of numbers which multiply together to get 16: {1, 16}, {2, 8}, and {4, 4}. These correspond to the 3 solutions in the problem (by adding 4 to each): {5, 20}, {6, 12}, and {8, 8}.

So now we solve a reframed problem: **How many pairs of integers are there that multiply together to get** $\mathbf{n^2}$. We would need the number of factors of $n^2$ in order to solve this. Since $n$ is a duplicated factor, the number of factors will be odd. 16 had 5 factors (1, 2, 4, 8, 16), which produced (5+1)/2 = 3 solutions.

To find the number of factors of $n^2$, we need the prime factorization of $n^2$. If the prime factorization of $n^2=p_1^{e_1}p_2^{e_2}\cdots p_k^{e_k}$, then the number of factors $f(n^2)$ is given by:

$$
f(n^2) = \prod_{i=1}^k (e_i + 1)
$$
This is true for any number, not just $n^2$. We add one to account for the duplicated factor, then divide 2, in other words, the number of solutions $S(n)$ is given by:

$$
S(n) = \frac{f(n^2) + 1}{2}
$$

We are given the number of solutions we need to find, and asked to find $n$. Our solution will be working backwards. If $S(n)> 1000$, then $f(n^2) \geq 2000$. To keep $n$ as small as possible, we must keep the exponents as small as possible. For $n^2$, we would want the exponents to only be either 2 or 4. In that case, when we calculate $f(n^2)$ the product will consist of 3s and 5s. Finding the smallest number with only these multiplicands will give us the smallest solution.

We want the **smallest number greater than 2000 that only has 3 and 5 as factors.** To generate them in order, we will keep two pointers $p_3$ and $p_5$ (0-indexed) that point to the previous number we multiplied by 3 or 5 respectively in the array $A$. At each step we compare $3A[p_3]$ and $5A[p_5]$ and add whichever product is smaller (and increment its corresponding pointer). If they're equal, we increment both. Here's an example to generate them in order:
* First, start with $A=[1]$, and $p_3=p_5=0$.
* Compare $3\times 1$ and $5\times 1$. The former is smaller, so we add 3 to the array and increment $p_3$. $A=[1,3]$ and $p_3=1$.
* Compare $3\times 3$ and $5\times 1$. The latter is smaller, so we add 5 and increment $p_5$. $A=[1,3,5]$ and $p_5=1$.
* $3\times 3 < 5\times 3$. Add 9 and increment $p_3$. $A=[1,3,5,9]$ and $p_3=2$.
* Here, $3\times 5 = 5\times 3$. Add 15 to the array and increment **both**. $A=[1,3,5,9,15]$ and $p_3=3$ and $p_5=2$.

In this fashion, we keep generating numbers until we go past our limit. We see the smallest number greater than 2000 is $3^4\times 5^2=2025$.

This factorization tells us that in $f(n^2)$, we are multiplying by 3 four times and 5 twice. Since these correspond to exponents, this means we have four prime squares, and two prime fourth powers in the prime factorization of $n^2$. Going a step further, in the factorization of $\mathbf{n}$, we have four prime singletons, and two prime squares.

To keep the value as small as possible, we assign larger powers to the smaller primes, and smaller powers to the larger primes. This means the minimum solution is $2^2\times 3^2\times 5\times 7\times 11\times 13 = \mathbf{180180}$.

The code is only to calculate the smallest number above 2000 that only contains 3 and 5.
```python
# file: "problem108.py"
factLimit = 2001
# Find smallest integer greater than
# factLimit that only has 3 and 5 as prime factors
# Two pointers to the current multiplicand
# for either 3 or 5.
threePoint = 0
fivePoint = 0
integers = [(1, [])]  # Start with 1 = 3^0 x 5^0
while integers[-1][0] < factLimit:
    # Find the minimum value of 3 multiplied
    # by the value at threePoint and 5 with
    # value at fivePoint. Increment the pointer
    # depending on which was chosen.
    # If they're both equal, then we add one of them
    # and increment both.
    if 3 * integers[threePoint][0] < 5 * integers[fivePoint][0]:
        integers.append((3 * integers[threePoint][0], integers[threePoint][1] + [3]))
        threePoint += 1
    elif 5 * integers[fivePoint][0] < 3 * integers[threePoint][0]:
        integers.append((5 * integers[fivePoint][0], integers[fivePoint][1] + [5]))
        fivePoint += 1
    else:
        integers.append((3 * integers[threePoint][0], integers[threePoint][1] + [3]))
        threePoint += 1
        fivePoint += 1

# Calculate value.
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
prod = 1
# Sort the powers greatest to least
for i, power in enumerate(sorted(integers[-1][1], key=lambda x: -x)):
    prod *= primes[i] ** (power - 1)
print(int(prod ** 0.5), 'with', (integers[-1][0] + 1) // 2, 'solutions.')
```
Running the code gets the correct output of,
```
180180 with 1013 solutions.
5.779999999999674e-05 seconds.
```
Thus, our final answer is **180180**.