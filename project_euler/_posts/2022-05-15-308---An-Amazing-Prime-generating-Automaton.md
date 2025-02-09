---
layout: post
title: "#308 - An Amazing Prime-generating Automaton"
date: 2022-05-15 11:07
number: 308
tags: [60_diff]
---

> A program written in the programming language Fractran consists of a list of fractions.
>
> The internal state of the Fractran Virtual Machine is a positive integer, which is initially set to a seed value. Each iteration of a Fractran program multiplies the state integer by the first fraction in the list which will leave it an integer.
>
> For example, one of the Fractran programs that John Horton Conway wrote for prime-generation consists of the following 14 fractions:
> $$
> \frac{17}{91}, \frac{78}{85}, \frac{19}{51}, \frac{23}{38}, \frac{29}{33}, \frac{77}{29}, \frac{95}{23}, \frac{77}{19}, \frac{1}{17}, \frac{11}{13}, \frac{13}{11}, \frac{15}{2}, \frac{1}{7}, \frac{55}{1}
> $$
> Starting with the seed integer 2, success iterations of the program produce the sequence:
>
> 15, 825, 725, 1925, 2275, 425, ..., 68, **4**, 30, ..., 136, **8**, 60, ..., 544, **32**, 240, ...
>
> The powers of 2 that appear in this sequence are $2^2$, $2^3$, $2^5$, ...
>
> It can be shown that *all* the powers of 2 in this sequence have prime exponents and that *all* the primes appear as exponents of powers of 2, in proper order!
>
> If someone uses the above Fractran program to solve Project Euler Problem 7 (find the $10001^{\text{st}}$ prime), how many iterations would be needed until the program produces $2^{10001\text{st prime}}$?

The best way to solve a problem like this is to work it out by hand. There is a good chance that a pattern will appear. In case it isn't clear, we employ the **first fraction from the left which results in an integer.** If we start with 2, then the first fraction which multiplies cleanly is $\frac{15}{2}$, and we are left with 15. Each step results in an integer, so there will always be a fraction whose denominator cleanly divides out. Let's label the fractions $A$ to $N$, from left to right.

| $A$             | $B$             | $C$             | $D$             | $E$             | $F$             | $G$             | $H$             | $I$            | $J$             | $K$             | $L$            | $M$           | $N$  |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | -------------- | --------------- | --------------- | -------------- | ------------- | ---- |
| $\frac{17}{91}$ | $\frac{78}{85}$ | $\frac{19}{51}$ | $\frac{23}{38}$ | $\frac{29}{33}$ | $\frac{77}{29}$ | $\frac{95}{23}$ | $\frac{77}{19}$ | $\frac{1}{17}$ | $\frac{11}{13}$ | $\frac{13}{11}$ | $\frac{15}{2}$ | $\frac{1}{7}$ | $55$ |

Notice the last "fraction" is just a whole number. To make this simple, I will show the prime factorization of each number, so we can clearly see the number's factors. Exponents will denote repeated multiplications e.g. $L^7$ means multiply by $L$ 7 times in a row, and $(EF)^4$ will mean multiply by $E$, then $F$, and repeat 4 times.

## Example from $2^3$ to $2^5$

Let's walk through an example where we see the multiplications that transform 8 to 32 in the following table.

| Integer                          | Multiplication                                               |
| -------------------------------- | ------------------------------------------------------------ |
| $2^3$                            | $L^3$                                                        |
| $3^3\times 5^3$                  | $N$                                                          |
| $3^3\times 5^4\times 11$         | $(EF)^3$ These two fractions together cancel out the 29, as long as we have a 3 |
| $11\times 7^3\times 5^4$         | $K$                                                          |
| $13\times 7^3\times 5^4$         | $(AB)^3$ $A$ cancels a 13 and 7, and introduces 17, while B cancels out that 17 along with a 5, while also introducing a 13 again, along with 3, and 2. |
| $13\times 5\times 3^3\times 2^3$ | $J$                                                          |
| $11\times 5\times 3^3\times 2^3$ | $(EF)^3$ Similar consequence as last time, the threes get converted into sevens |
| $11\times 7^3\times 5\times 2^3$ | $K$                                                          |
| $13\times 7^3\times 5\times 2^3$ | $AB$ There is only one 5, so we can only do this pair multiplication once |
| $13\times 7^2\times 3\times 2^4$ | $A$                                                          |
| $17\times 7\times 3\times 2^4$   | $C$                                                          |
| $19\times 7\times 2^4$           | $(DG)^4$ We introduce a 23, which is canceled by $G$, which introduces a 19 again, along with a 5. Thus, the twos are converted into fives |
| $19\times 7\times 5^4$           | $H$                                                          |
| $11\times 7^2\times 5^4$         |                                                              |

Notice the last line and how it compares to line 4 ($11\times 7^3\times 5^4$). We have reduced the power on the 7 by one through a series of multiplications. Thus, we can speed things up a little bit:

| Integer                            | Multiplication                                               |
| ---------------------------------- | ------------------------------------------------------------ |
| $11\times 7^2\times 5^4$           | $K(AB)^2J(EF)^2$ The exponent is controlled by the number of 7s we have |
| $11\times 7^2\times 5^2\times 2^2$ | $K$                                                          |
| $13\times 7^2\times 5^2\times 2^2$ | $(AB)^2$                                                     |
| $13\times 3^2\times 2^4$           | $J$                                                          |
| $11\times 3^2\times 2^4$           | $(EF)^2$                                                     |
| $11\times 7^2\times 2^4$           | $K$                                                          |
| $13\times 7^2\times 2^4$           | $A$                                                          |
| $17\times 7\times 2^4$             | $I$                                                          |
| $\mathbf{7\times 2^4}$             |                                                              |

Notice now that after another "iteration" (with few changes) we have a power of 2. However, this is paired with a power of 7. We will see that when we encounter a power of 2 in the sequence, if it is **not** a prime power of 2, it will paired with some power of 7. Let's continue! We do a repeat of the steps we started with (except now we use $M$ due to the 7), and so I'll condense it even more. Pay close attention to the powers and how the relate to the previous row.

| Integer                  | Multiplication                                |
| ------------------------ | --------------------------------------------- |
| $7\times 2^4$            | $L^4$                                         |
| $7\times 5^4\times 3^4$  | $M$                                           |
| $5^4\times 3^4$          | $N$                                           |
| $11\times 5^5\times 3^4$ | $(EF)^4$                                      |
| $11\times 7^4\times 5^5$ | $K(AB)^4J(EF)^4K(AB)AC(DG)^5H$                |
| $11\times 7^3\times 5^5$ | $K(AB)^3J(EF)^3K(AB)^2AC(DG)^5H(EF)$          |
| $11\times 7^2\times 5^5$ | $\left(K(AB)^2J(EF)^2\right)^2K(AB)AC(DG)^5H$ |
| $11\times 7\times 5^5$   | $\left( K(AB)J(EF) \right)^5KAI$              |
| $\mathbf{2^5}$           |                                               |

## Recap and Observations

It definitely appears that there is some level repetition going on, and the repeats depend on the exponents in the expression $11\times 7^a\times 5^b$. When our number looks like this, we perform $K(AB)^aJ(EF)^a$ a certain number of times, followed by $K(AB)^cAC(DG)^bH$, where $c=b\mod a$. However, sometimes, that second step is not necessary, as if $a$ and $b$ are "nice", we can do $KAI$ instead, and are left with $7^d\times 2^p$. If $p$ happens to be prime, then $d=0$.

## A Longer Sequence

Through these observations, we can conclude that the number of multiplications in the sequence between one form of $7^d\times 2^p$ to another is deterministic, but we need a slightly longer multiplication to see the full process. Let us start from $2^7$! I will speed things up considerably, and step between numbers of the form $11\times 7^a\times 5^b$. Pay attention to how we many times me multiply each set of fractions and how it relates to the exponents in the number. 

| Integer                        | Multiplication                                           |
| ------------------------------ | -------------------------------------------------------- |
| $2^7$                          | $L^7N(EF)^7$                                             |
| $11\times 7^7\times 5^8$       | $K(AB)^7J(EF)^7K(AB)AC(DG)^8H$                           |
| $11\times 7^6\times 5^8$       | $K(AB)^6J(EF)^6K(AB)^2AC(DG)^8H(EF)$                     |
| $11\times 7^5\times 5^8$       | $K(AB)^5J(EF)^5K(AB)^3AC(DG)^8H(EF)^2$                   |
| $11\times 7^4\times 5^8$       | $\left(K(AB)^4J(EF)^4\right)^2KAI$                       |
| $7^3\times 2^8$                | $L^8M^3N(EF)^8$                                          |
| $11\times 7^8\times 5^9$       | $K(AB)^8J(EF)^8K(AB)AC(DG)^9H$                           |
| $11\times 7^7\times 5^9$       | $K(AB)^7J(EF)^7K(AB)^2AC(DG)^9H(EF)$                     |
| $11\times 7^6\times 5^9$       | $K(AB)^6J(EF)^6K(AB)^3AC(DG)^9H(EF)^2$                   |
| $11\times 7^5\times 5^9$       | $K(AB)^5J(EF)^5K(AB)^4AC(DG)^9H(EF)^3$                   |
| $11\times 7^4\times 5^9$       | $\left(K(AB)^4J(EF)^4\right)^2K(AB)AC(DG)^9H$            |
| $11\times 7^3\times 5^9$       | $\left(K(AB)^3J(EF)^3\right)^3KAI$                       |
| $7^2\times 2^9$                | $L^9M^2N(EF)^9$                                          |
| $11\times 7^9\times 5^{10}$    | $K(AB)^9J(EF)^9K(AB)AC(DG)^{10}H$                        |
| $11\times 7^8\times 5^{10}$    | $K(AB)^8J(EF)^8K(AB)^2AC(DG)^{10}H(EF)$                  |
| $11\times 7^7\times 5^{10}$    | $K(AB)^7J(EF)^7K(AB)^3AC(DG)^{10}H(EF)^2$                |
| $11\times 7^6\times 5^{10}$    | $K(AB)^6J(EF)^6K(AB)^4AC(DG)^{10}H(EF)^3$                |
| $11\times 7^5\times 5^{10}$    | $\left(K(AB)^5J(EF)^5\right)^2KAI$                       |
| $7^4\times 2^{10}$             | $L^{10}M^4N(EF)^{10}$                                    |
| $11\times 7^{10}\times 5^{11}$ | $K(AB)^{10}J(EF)^{10}K(AB)AC(DG)^{11}H$                  |
| $11\times 7^9\times 5^{11}$    | $K(AB)^9J(EF)^9K(AB)^2AC(DG)^{11}H(EF)$                  |
| $11\times 7^8\times 5^{11}$    | $K(AB)^8J(EF)^8K(AB)^3AC(DG)^{11}H(EF)^2$                |
| $11\times 7^7\times 5^{11}$    | $K(AB)^7J(EF)^7K(AB)^4AC(DG)^{11}H(EF)^3$                |
| $11\times 7^6\times 5^{11}$    | $K(AB)^6J(EF)^6K(AB)^5AC(DG)^{11}H(EF)^4$                |
| $11\times 7^5\times 5^{11}$    | $\left(K(AB)^5J(EF)^5\right)^2K(AB)AC(DG)^{11}H$         |
| $11\times 7^4\times 5^{11}$    | $\left(K(AB)^4J(EF)^4\right)^2K(AB)^3AC(DG)^{11}H(EF)^2$ |
| $11\times 7^3\times 5^{11}$    | $\left(K(AB)^3J(EF)^3\right)^3K(AB)^2AC(DG)^{11}H(EF)$   |
| $11\times 7^2\times 5^{11}$    | $\left(K(AB)^2J(EF)^2\right)^5K(AB)AC(DG)^{11}H$         |
| $11\times 7\times 5^{11}$      | $\left(K(AB)J(EF)\right)^{11}KAI$                        |

Now it is much more clear how the repetitions come about. $11\times 7^b\times 5^{b+1}$. The following conclusions can be drawn:

- If we start with $7^a\times 2^b$, then a repetitive pattern forms starting with $11\times 7^b\times 5^{b+1}$. 
- Let our current product look like $11\times 7^x\times 5^y$. A **repetitive** step is performed, and the multiplication is of the form $\left(K(AB)^xJ(EF)^x\right)^{\lfloor y/x \rfloor}K(AB)^{y\mod x}AC(DG)^yH(EF)^{(y\mod x) - 1}$.
- However, if $p$ is the **greatest number divisible by $y$**, then a **reductive** step is performed, and the multiplication is of the form $\left(K(AB)^pJ(EF)^p\right)^{y/p}KAI$.
- If we have a number of the form $7^a\times 2^b$, then the multiplication is of the form $L^bM^aN(EF)^b$.

Let's take a closer look at the **repetitive** step. Notice that, even though the number of repetitions of different parts are changing, that the number of multiplications from one row to the next are constant! For example, for $11\times 7^b\times 5^{11}$, it is always *70* multiplications each time.

One intuitive reason that it always stays constant is because $\left\lfloor \frac{y}{x} \right\rfloor$ and $y\mod x$ are linked. As $x$ increases/decreases, both increase and decrease by the same moment, and so any changes in the individual expressions are cancelled out perfectly. In our multiplication expression, there are two places that rely on $\left\lfloor \frac{y}{x} \right\rfloor$, and two places that rely on $y\mod x$. If the first number for the repetitive step is $11\times 7^b\times 5^{b+1}$, then the number of multiplications in each repetitive step is $6+2(b+b+b+1+1) = 6+2(3b+2) = \mathbf{6b+10}$.

We have all the pieces we need to come up with a deterministic formula.

## Deterministic Formula

The formula we develop will calculate the number of multiplications needed to go from a number of the form $7^a\times 2^b$ to the next. If $a=0$, then $b$ is prime.

Let our integer be $7^{x_1}\times 2^y$, and $p$ be the greatest number divisible by $y$. Then the next number that looks like this will be $7^{x_2}\times 2^{y+1}$, where $x_2 = \frac{y+1}{p}-1$. The technical number of multiplications needed to get there is as follows:

$$
m = \underbrace{y+x_1+1+2y}_{L^yM^{x_1}N(EF)^y}+\sum_{i=p+1}^y\left( \left\lfloor\frac{y+1}{i}\right\rfloor\underbrace{(1+2i+1+2i)}_{K(AB)^iJ(EF)^i} + \underbrace{1+2((y+1)\mod i)+2}_{K(AB)^{(y+1)\mod i}AC}+\underbrace{2(y+1)+1+2((y+1)\mod i-1)}_{(DG)^{y+1}H(EF)^{(y+1)\mod i-1}} \right)+\underbrace{\frac{y+1}{p}(1+2p+1+2p)+3}_{(K(AB)^pJ(EF)^p)^{\frac{y+1}{p}}KAI}
$$
However, using our observations in the previous section, we can substitute the entire summation with a single $6y+10$. Thus, we can simplify as follows:
$$
\begin{aligned}
m&=3y+x_1+1+\sum_{i=p+1}^y (6y+10)+(y+1)\left(\frac{2}{p}+4\right)+3
\\ &=
3y+x_1+1+(y-p)(6y+10)+(y+1)\left(\frac{2}{p}+4\right)+3
\end{aligned}
$$


While we can simplify a bit further, it is not needed. Writing in the above manner allows us to clearly see the different parts of the process.

## Code and Implementation

We have all the pieces to officially code this up. The first order of business is calculating $p$. The 10001st prime is a tad over 100000, so we shouldn't take too long in our looping. The value $p$ is the **largest number that is divisible by $y$** (not including $y$ itself). Put another way, $p$ is the result you get when dividing $y$ by its **smallest prime factor**. Thus, if we know the smallest prime factor, we instantly know $p$. 

We can save the smallest prime factor for all numbers using a sieve-like approach. Then, in our loop, we can simply grab the correct $p$, and instantly calculate the multiplications needed.

```python
import numpy as np
import primesieve

# Create a "sieve" where each element represents the greatest
# number divisible by the index.
LIMIT = 10001
primes = primesieve.n_primes(LIMIT)
sieve = np.zeros(primes[-1] + 1, dtype=int)
for p in primes[::-1]:
    sieve[::p] = p
sieve[1] = 1  # To prevent divide by zero error, we won't actually use this value...
sieve = np.arange(primes[-1] + 1) // sieve

# We start at 2^1. A prime power of 2, will NOT have a power of 7 (x) with it.
# So this is when we know to increment the prime counter...
prime_count = 0
y = 1
total_steps = 0
x = 0

for y, P in enumerate(sieve[2:], 1):
    # Calculate the multiplications that convert to 11 x 7^(..) x 5^(..)
    steps = y + x + 1 + 2 * y
    # Calculate the repeated multiplications
    steps += (y - P) * (10 + 6 * y)
    # Calculate the conversion to a power of 2
    steps += (y + 1) // P * (4 * P + 2) + 3

    # Calculate the resulting power of 7...
    x = P - 1

    total_steps += steps


print('Prime #{} is {} and it took {} steps to get there.'.format(LIMIT, y, total_steps))
```

Running this short loop, we get

```
Prime #10001 is 104742 and it took 1539656383221924 steps to get there.
0.07477193200003285 seconds.
```

Thus, the total number of steps needed to reach the 10001st prime is **1539656383221924**. This problem required a lot of pen and paper and doing operations by hand in order to see the correct pattern necessary.
