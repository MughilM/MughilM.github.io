---
layout: post
title: "#233 - Lattice points on a circle"
date: 2019-07-20 16:10
number: 233
tags: [70_diff]
---
> Let $f(N)$ be the number of points with integer coordinates that are on a circle passing through $(0,0),(N,0),(0,N)$, and $(N,N)$.
> 
> It can be done shown that $f(10000)=36$.
> 
> What is the sum of all positive integers $N\leq 10^{11}$ such that $f(N)=420$?
{:.lead}
* * *

This is one of my favorite problems that I've solved for. As for the circle, it passes through 4 points of a square with side length $N$. Since it circumscribes the square the diameter of the circle is the diagonal of the square, namely $N\sqrt{2}$. The radius is then $N\sqrt{2}/2$. Both the circle and square also have the same center. Therefore, the center of the circle is $(N/2, N/2)$. So, our equation for this circle is 

$$
\left(x-\frac{N}{2}\right)^2+\left(y-\frac{N}{2}\right)^2=\frac{N^2}{2}
$$

The solution is long, and before I dive in, you should watch the following video, as it gives extremely useful information in solving the problem. It is by Grant Sanderson of the YouTube channel 3Blue1Brown.
[![Circle](https://img.youtube.com/vi/NaL_Cb42WyY/0.jpg)](https://www.youtube.com/watch?v=NaL_Cb42WyY)

To summarize the key points:
* The number of lattice points that a circle centered at the origin with radius $\sqrt{N}$ crosses is dependent on the **prime factorization** of $N$.
* Write the factorization of $N$ as $2^{a_0}\left( 5^{b_1} 13^{b_2}\cdots\right)\left( 3^{c_1}7^{c_2}\cdots \right)$ where the primes associated with $b_i$ are **one above a multiple of 4** and the primes associated with $c_i$ are **one below a multiple of 4**.
* The number of lattice points is $4(b_1+1)(b_2+1)\cdots=4\prod_{i=1}^\infty(b_i+1)$
* If any $c_i$ is odd, then the number of lattice points is 0. Otherwise, they do not affect the count.
* Factors of 2 do not affect the count.

This is all well and good, but there is one catch with this problem.

## Grant's Circle vs. Our Circle
The circle in the video is centered at the origin and has radius $\sqrt{N}$, while our circle doesn't have that center, nor that radius. If $N$ is even, then our circle lands at a lattice point, which we can translate back to the origin and compute that way. However, if $N$ is odd, then it doesn't land a lattice point. 

But I claim that we can transform our existing problem into exactly the one shown in the video. In other words, the two circles

$$
\begin{aligned}
    \left(x-\frac{N}{2}\right)^2 + \left(y-\frac{N}{2}\right)^2 &= \frac{N^2}{2}
    \\
    x^2+y^2 &= N^2
\end{aligned}
$$
will cross the **same number of lattice points**. We break this up into cases, either $N$ is even, or $N$ is odd.
### $N$ is even
$N$ can be represented as $2k$, where $k$ is any integer. Substituting into our original circle, we get

$$
\begin{aligned}
    \left(x-\frac{2k}{2}\right)^2 + \left(y-\frac{2k}{2}\right)^2 &= \frac{(2k)^2}{2}
    \\
    (x-k)^2+(y-k)^2 &= 2k^2
\end{aligned}
$$

This is a circle centered at $(k,k)$, which is in effect can be our "origin". Next, remember that factors of 2 do not affect the count. Thus, a circle with radius $\sqrt{2k^2}$ will cross the same number of lattice points as one with radius $\sqrt{2\cdot2k^2}$. But $\sqrt{2\cdot2k^2} = \sqrt{4k^2}=\sqrt{(2k)^2}=\sqrt{N^2}=N$. Therefore, we can make the radius of the circle be $N$ and not effect the count.
### $N$ is odd
If $N$ is odd, then the right-hand side (RHS) of the eqution will stay as a fraction. This means the square on the left-hand side (LHS) must also be a fraction. We can then assume something like this:

$$
\begin{aligned}
\left( x-\frac{N}{2} \right)^2 &= \frac{a^2}{4}
\\
\left( y-\frac{N}{2} \right)^2 &= \frac{b^2}{4}
\end{aligned}
$$

where $a$ and $b$ are integers. Adding these, our equation becomes

$$
\begin{aligned}
    \frac{a^2}{4} + \frac{b^2}{4} &= \frac{N^2}{2}
    \\
    a^2+b^2 &= 2N^2
\end{aligned}
$$

The last equation is a circle centered on the origin. Additionally, factors of 2 don't affect count, which means this crosses the same amount of lattice points.

## Working Through the Problem's Example
We have convinced ourselves that the problem circle will cross the same amount of lattice points as the one that is centered on the origin with radius $N$. I'll quickly work through the given example in the problem, where $N=10000$. We need to factorize the radius $10000^2 = 10^8$.

The video was dealing with a circle with radius $\sqrt{N}$ and he factorized $N$. Here, we have a radius of $N$ and so we must factorize $N^2$
{:.note}

The prime factorization of $N^2=10^8$ is $2^8\times 5^8$. Factors of 2 don't affect the count. 5 can be broken down into complex factors. Add one to the exponent, then multiply by 4. Therefore, the total is 36, just as the problem.

Each exponent of the factorization of $N^2$ will be **even**. If Gaussian primes (3, 7, etc.) have even exponents, then _that does not_ affect the count. This fact becomes important when we find a way to solve the problem.
## Solving the Problem
This problem is actually asking for the **inverse** of what we just did. Given the number of points $P$, what values of $N$ whose circles cross that number of points?

We can work backwards. If we know $P$, we can divide by 4, and find the factorization of $P$. For example, in the problem, the anwser is 36. Divide by 4 and we get 9. This means in the prime exponents, we have either **one 8**, or **two exponents that are 2 (3 times 3 = 9)**. We can include another non-Gaussian prime in our factorization, and so another solution to this is $2^4\times 5^2\times 13^2=67600=260^2$. Since Gaussian primes are squared and don't affect the count, here's another solution: $2^2\times 3^2\times 5^2\times 7^2\times 17^2 = 12744900 = 3570^2$. To get all values of $N$ below a limit $L$, we need to mix and match the exponents of the non-Gaussian primes (5, 13, 17, etc.) with the exponents of 2 and the Gaussian primes.

Here's another we look at the example. Since we need the exponents in $N^2$ to either be 8 or two 2s, the exponents in $\mathbf{N}$ need to be either 4 or two 1s (because we are squaring which doubles the exponents). We're almost at the point where we can get an algorithm to search all $N$ up to $L$.

### Example of $N=36$ and $L=1000$
We have two separate cases, either a single non-Gaussian prime raised to the 4th power, or two separate singleton primes. The smallest prime that can affect the count is 5. Only $5^4=625<1000$, so that's the only value below our limit for the 4th power case.

The next case is two singleton primes. Since the smallest (non-Gaussian) prime is 5, the other factor needs to be less than 1000/5 = 200. The next is 13, so the smallest $N$ that crosses 36 lattice points is $5\times 13=65$.

But we can also take all the other Gaussian primes (3, 7, etc.) in addition 2 and include them in the product as well e.g. $2\times3\times5\times13=390$, $3^2\times5\times13=585$, and $2\times7\times5\times13=910$ all cross 36 lattice points. We have a separate set of products that we can multiply with $5\times 13$ in order to get $N$s that cross 36 points.

This other product can't exceed $\lfloor 1000/65 \rfloor=15$. We can multiply by any number between 2 and 14, **except 5 and 10, as those are multiples of 5 and therefore affect the count.** Multiplying 65 by each of these numbers will get additional $N$s that cross 36 points.

Once we're done with 65, we move to $5\times 17=85$ and repeat the process.

## Algorithm
Given the number of crossed lattice points $C$ and the limit $L$, our algorithm is as follows:
* Divide $C$ by 4, and find **all possible products** that will result in $C/4$. Subtract one and halve the values to obtain the exponent values for $N$.
* For each set of exponents, get all possible products that _only_ consist of non-Gaussian primes ($1\mod 4$) below $L$. Let this set be $P_1$.
* For each product $r$ in $P_1$, generate a second set of products $P_2$ less than $L/r$ whose factorizations only consist of 2 and primes that are $3\mod 4$.
* Multiply $r$ by each product in $P_2$ to obtain an $N$ value less than $L$ that crosses exactly $C$ lattice points. 

## Implementation
Our number is 420. Divide by 4 and we have **105**. The prime factorization of 105 is $3\times 5\times 7$. Now we must find all possible values of the exponents.
* One is cubed, one is squared, and the other is a singleton: $\{3,2,1\}$ (Taken directly from the 3, 5, and 7. Subtract one and divide by 2).
* One is raised to the 10th power, and the other is squared: $\{10, 2\}$ (This comes from $21\times 5 = 105$)
* One is raised to the 7th power, and the other is cubed: $\{7, 3\}$ (This comes from $15\times 7=105$) 

We will need to loop through each case, as well as a list of $1\mod 4$ primes and products of $3\mod 4$ primes (including 2). We only need primes until $\frac{10^{11}}{5^313^2}=4\,733\,727$, because our smallest solution is $5^3\times13^2\times17=359125$. 

For each power case, we reduce the limit by the prime we select raised to that power. The most constraining case of a 10th and 7th power will result in the limit reducing fairly quickly. This theme is present for all the loops we have (e.g. given a chosen prime $p$ for the 10th power, the next prime for the 7th power has to be less than $\sqrt[7]{L/p^{10}}$. The three loops and full code is below.
```python
# file: "problem233.py"
limit = 10 ** 11
# Maximum prime list we need
# is up to 10^11 / (5^3 * 13^2)
primeBound = limit / (125 * 169)
primes = primesieve.numpy.primes(primeBound)
# Grab all primes that are of 1 mod 4
primes = primes[primes % 4 == 1]

# Now the sieve of all integers that
# HAVE at least one factor from the prime list.
# The max size possible should be
# 10^11 / (5^3 * 13^2 * 17), since that is our
# smallest solution.
sieveLimit = int(primeBound / 17)
sieve = np.zeros(sieveLimit + 1, dtype=np.uint8)
for prime in primes:
    # Grab multiples
    multiples = np.arange(prime, sieveLimit + 1, prime)
    # Set all locs to one...
    sieve[multiples] = 1

# Cases 1 and 2, powers of 10 and 2, and powers of 7 and 3
total = 0
for ex1, ex2 in [(10, 2), (7, 3)]:
    for p1 in primes[primes < limit ** (1/ex1)]:
        # Now we need all the different primes
        # that are less than the square/cube root
        # of 10^11 / prime^10
        for p2 in primes[(primes < (limit / p1 ** ex1) ** (1/ex2)) & (primes != p1)]:
            value = p1 ** ex1 * p2 ** ex2
            numOfMultiplies = limit // value
            # Get non 1 mod 4 prime multiples, multiply them,
            # and finally sum them...
            # DON'T INCLUDE 0...
            multiples = np.where(sieve[:numOfMultiplies + 1] == 0)[0][1:]
            total += np.sum(value * multiples)

# Case 3, where we have THREE primes,
# one cubed, squared, and the remaining just multiplied.
for p1 in primes[primes < limit ** (1/3)]:
    # Next is squared...
    for p2 in primes[(primes < (limit / p1 ** 3) ** (1/2)) & (primes != p1)]:
        # Next get all individually...
        for p3 in primes[(primes < limit / (p1 ** 3 * p2 ** 2)) & (primes != p1) &
                         (primes != p2)]:
            value = p1 ** 3 * p2 * p2 * p3
            numOfMultiplies = limit // value
            multiples = np.where(sieve[:numOfMultiplies + 1] == 0)[0][1:]
            total += np.sum(value * multiples)

print(total)
```
Finally, running this monster code gives us
```
271204031455541309
7.3993970000000004 seconds.
```
Therefore, the sum of all $N<10^{11}$ where the circle passes through exactly 420 lattice points is **271204031455541309**. A massive number for a problem that took a massive amount of time.
