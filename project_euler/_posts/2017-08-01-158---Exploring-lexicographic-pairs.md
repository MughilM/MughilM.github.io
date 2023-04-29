---
layout: post
title: "#158 - Exploring strings for which only one character comes lexicographically after its neighbour to the left"
date: 2017-08-01 15:31
number: 158
tags: [55_diff]
---
> Taking three different letters from the 26 letters of the alphabet, character strings of length three can be formed.
>
> Examples are 'abc', 'hat' and 'zyx'.
>
> When we study these three examples we see that for 'abc' two characters come lexicographically after its neighbour to the left. For 'hat' there is exactly one character that comes lexicographically after its neighbour to the left. For 'zyx' there are zero characters that come lexicographically after its neighbour to the left. 
>
> In all there are 10400 strings of length 3 for which exactly one character comes lexicographically after its neighbour to the left.
>
> We now consider strings of $n\leq 26$ different characters from the alphabet. For every $n$, $p(n)$ is the number of strings of length of $n$ for which exactly one character comes lexicographically after its neighbour to the left.
>
> What is the maximum value of $p(n)$?
{:.lead}
* * *

This is a counting problem. To put their restriction another way, **there is only one pair of letters which are in order** This also tells us that the substrings on either side of the pair are in decreasing order.

For example, with 5 letters: $c, t, m, r, d$, a satisfying string is $tmcrd$. The pair in order is $cr$, and both $tmc$ and $rd$ are in decreasing order. 

Given a group of $n$ letters, we can find a formula for the number of strings satisfying the property $c(n)$. This is not $p(n)$, as that value takes all possible groups. We will aggregate later.

## Finding a formula for $c(n)$
Let's assume our group of $n$ letters be the first $n$ letters of the alphabet. There are some trivial values for the lower values of $n$: 
* $c(0) = c(1) = 0$, because it is impossible to order anything lower than 2 letters.
* $c(2) = 1$, as the pair is either in order, or out of order.
* $c(3) = 4$. Those strings are $abc$, $acb$, $bca$, and $bac$.
* $c(4)=11$. We have a couple of cases here for where $d$ can be placed.
    * We can prepend $d$ to all the satisfactory string and get 4 new satisfactory strings: $dabc$, $dacb$, $dbca$, $dbac$.
    * If $d$ is in position 2, the 1st letter needs to be less than $d$. All of $a$, $b$, and $c$ are, so this produces 3 new strings: $adcb$, $bdca$, $cdba$
    * If $d$ is in position 3, then the first two letters need to be in reversed order. We can choose 2 letters from the 3 in 3 ways to produce 3 strings: $badc$, $cadb$, $cbad$.
    * If $d$ is last, then the first 3 letters need to be reversed, of which there is only 1: $cbad$.
Please work through $c(5)$ to get a feel for the pattern. For a given $n$:
* Prepend the $n^{\text{th}}$ letter to each of the strings previously. This is $c(n-1)$.
* For the $i^{\text{th}}$ position i the string ($i\geq 2$), the $i-1$ letters before it need to be reversed. We can choose the letters from the remaining group a total of $\binom{n-1}{i-1}$ ways, and produces as many strings.
* Repeat for all $i$ and add up all values (from 2 to $n$).

In conclusion,

$$
\begin{aligned}
c(n) &= \binom{n-1}{1}+\binom{n-1}{2}+\cdots+\binom{n-1}{n}+c(n-1) 
\\ &= \sum_{i=1}^n\binom{n-1}{i}+c(n-1) 
\\ &= \boxed{2^{n-1}-1+c(n-1)}
\end{aligned}
$$
## Finding a formula for $p(n)$
Now that we have $c(n)$, finding a formula for $p(n)$ is actually simple. Although we demonstrated with the first $n$ letters, the group of letters actually does not matter for the value of $c(n)$. The value of $p(n)$ takes into account the value of $c(n)$ for **all** groups of $n$ letters. There are $\binom{26}{n}$ different groups. Therefore,

$$
p(n)=\binom{26}{n}c(n)
$$
## Implementation
The recursive nature of $c(n)$ allows us to precompute values of $c(n)$ beforehand. We create an array of values of $p(n)$ and check the maximum afterwards. I use the `math.factorial` to compute the binomial coefficient.
```python
# file: "problem158.py"
def binom(n, r):
    return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

values = [0]
for n in range(1, 27):
    values.append(2 ** (n - 1) - 1 + values[-1])
# Now multiply each by 26 choose n
values = [int(value * binom(26, i)) for i, value in enumerate(values)]
print('Max value of', np.max(values), 'at n =', np.argmax(values))
```
Running our loop get us,
```
Max value of 409511334375 at n = 18
0.0015439999988302588 seconds.
```
Therefore, the maximum value of $p(n)$ is **409511334375**.