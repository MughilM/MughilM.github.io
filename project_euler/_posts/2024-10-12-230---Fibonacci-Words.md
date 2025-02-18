---
layout: post
title: "#230 - Fibonacci Words"
date: 2024-10-12 21:13
number: 230
tags: [50_diff]
---

> For any two strings of digits, $A$ and $B$, we define $F_{A,B}$ to be the sequence $(A,B,AB,BAB,ABBAB,\dots)$ in which each term is the concatenation of the previous two. 
>
> Further, we define $D_{A,B}(n)$ to be the $n^{\text{th}}$ digit in the first term of $F_{A,B}$ that contains at least $n$ digits.
>
> Example:
>
> Let $A = 1415926535$, $B = 8979323846$. We wish to find $D_{A,B}(35)$, say.
>
> The first few terms of $F_{A,B}$ are:
> $$
> 1415926535 \\
> 8979323846 \\
> 14159265358979323846 \\
> 897932384614159265358979323846 \\
> 1415926535897932384689793238461415\mathbf{\color{red}9}265358979323846
> $$
> 
>
> Then $D_{A,B}(35)$ is the $35^{\text{th}}$ digit in the fifth term, which is 9.
>
> Now we use for $A$ the first 100 digits of $\pi$ behind the decimal point:
>
> $14159265358979323846264338327950288419716939937510$
>
> $58209749445923078164062862089986280348253421170679$
>
> and for $B$ the next hundred digits:
>
> $82148086513282306647093844609550582231725359408128$
>
> $48111745028410270193852110555964462294895493038196$
>
> Find $\sum_{n=0}^{17}10^n\times D_{A,B}((127+19n)\times 7^n)$

## Background

In the summations, if $n=17$, we have a very large argument for $D_{A,B}$. The string that is generated until we have that many digits will be too large to keep in memory. Because these are "Fibonacci" words, the number of individual strings i.e. $A$'s and $B$'s that are present at each iteration will grow according to Fibonacci words.

The sequence $A,B,AB,BAB,ABBAB,BABABBAB,\dots$ goes $1, 1, 2, 3, 5, 8$.

## Methodology

The main question is as follows: **Do we need to compute the strings explicitly?** Is it possible to find $D_{A,B}(n)$ without computing a string that is at least $n$ digits long? Here's an observation. In the smaller example, both $A$ and $B$ were 10 digits long. When computing $D_{A,B}(35)$, the fifth string was 50 digits, so a $35^{\text{th}}$ digit existed. It was 50 digits long, because the fifth term in the Fibonacci sequence was 5, and $5\times 10 = 50$. 

The fifth Fibonacci word is $ABBAB$. Since we need the $35^{\text{th}}$ digit, and each string is 10 digits long, this means the **fifth entry in the fourth string** is our answer. The fourth string is $A$ and and the fifth digit is 9, our answer.

Ok, so it doesn't seem like we need to compute entire string explicitly. However, we can actually go one step further, and not even need the strings of $A$'s and $B$'s!

### Working backwards

We can essentially use the Fibonacci formula in reverse. Again, using our example, we know we need the fifth Fibonacci word, which corresponds to the Fibonacci number of 5. However, we know $5=2+3$ (the previous two Fibonacci numbers, and hence the previous two Fibonacci words). Thus, the first 20 digits is one "word", and the next 30 digits is another "word". Because $35>20$, the $35^{\text{th}}$ digit of the 50-digit word, is actually the $15^{\text{th}}$ digit of the 30-digit word (as $50-35=15$).

We just made our problem smaller. Instead of needing the $35^{\text{th}}$ digit of a $50$-digit word, we now need the $15^{\text{th}}$ of a $30$-digit word. Let's repeat one more step. $30$ corresponds to the Fibonacci number of $3$, and $3=1+2$. We have a $10$-digit word (which is $B$ in this case) attached to a $20$-digit word ($AB$). Next, $15>10$, and $20-15=5$, so we need the $5^{\text{th}}$ digit of $AB$. This is easy, as it is just the $5^{\text{th}}$ of $A$, which is $9$, our answer.

Using this "reduction" method, we were able to completely reduce the problem into finding a digit in the original strings of $A$ and $B$. It was relatively simple to convert the lengths to their corresponding Fibonacci numbers, because $A$ and $B$ were the same length, but you can make a similar argument for different length strings. Thankfully, in the larger problem, $A$ and $B$ are still the same length. 

The only necessary calculation we need is a list of all the Fibonacci numbers. However, Fibonacci numbers grow exponentially, and so we shouldn't need too many. We can use its explicit formula to find how many we need.

## Code and Implementation

The code is relatively straight forward, and I chose an iterative approach to the reduction. For a recursive approach, you would need to keep track of which Fibonacci number your string refers to, and the base case would correspond to having an index which is less than the length of the original string.

In our function, we take $n$, $A$, $B$, and a list of Fibonacci numbers that are big enough. I also include a small one-off function that calculates the largest Fibonacci number we need.

```python
def fib_index(n):
    return int((math.log(n) + math.log(5) / 2) / math.log((1 + math.sqrt(5)) / 2)) + 1

def dab(n, A, B, fibs: list):
    """
    Finds D_a,b(n) using the Fibonacci word expansion.
    """
    # There's no need to keep computing the next Fibonacci string
    # until the string is long enough. Since each string is built from A
    # and B, we will cut-down the needed string length, basically computing
    # Fibonacci in reverse until we reach either A or B.
    # For example, len(A) = len(B) = 10, and n = 35, this means the fifth Fibonacci
    # expansion has at least 50 characters (corresponding to F value of 5).
    # 5 = 2 + 3, since 35 >= 20, the character we need is in the last 30 charas,
    # and it would be the 15th chara. 3 = 1 + 2, again it's in the second part (20 - 15 = 5th chara).
    # 2 = 1 + 1 (A + B), 5 < 10, so the 5th character of A is 9, which is the answer.
    # One issue, we need the precomputed Fibonacci numbers...
    # ASSUME A AND B HAVE THE SAME LENGTH.
    if n <= 10:
        return A[n - 1]
    str_length = len(A)
    # Find where we start at...
    i = 0
    while fibs[i] * str_length < n:
        i += 1
    # Keep going until we land at one of the starting two strings (either index 0 or index 1 in fibs)
    while i >= 2:
        # Find which part of the string we need to jump to, by comparing the previous two
        # Fibonacci values and the current value of n.
        if n > fibs[i - 2] * str_length:
            # We want i - 1, adjust n as necassary.
            n -= fibs[i - 2] * str_length
            i -= 1
        else:
            # We want i - 2, in this case, n doesn't change
            i -= 2
    # Take whichever string we need...
    return int(A[n - 1]) if i == 0 else int(B[n - 1])
```

All that's left is to calculate our summation using this function.

```python
A = '1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'
B = '8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196'
# Using the max possible value for n, we can find what Fibonacci number we calculate up to.
# Divide n by the length of A or B.
L = 17
MAX_N = (127 + 19 * L) * 7 ** L
MAX_FIB = MAX_N // 100
fval = fib_index(MAX_FIB) + 1
print(f'Need {fval} Fibonacci numbers.')
# Precompute Fibonacci numbers...
fibs = [1, 1]
for _ in range(fval - 2):
    fibs.append(fibs[-1] + fibs[-2])
print(fibs)

s = sum(10 ** n * dab(7 ** n * (127 + 19 * n), A, B, fibs) for n in range(L + 1))

print(s)
```

Running this, we get

```
Need 75 Fibonacci numbers.
850481152593119296
0.00019055799998568546 seconds.
```

Therefore, the value of the summation is **850481152593119296**, and it turns out we needed 75 Fibonacci numbers (up to 2111485077978050).

