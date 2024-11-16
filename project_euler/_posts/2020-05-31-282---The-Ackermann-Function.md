---
layout: post
title: "#282 - The Ackermann Function"
date: 2020-05-31 13:00
number: 282
tags: [70_diff]
---
> For non-negative integers $m$, $n$, the Ackermann function $A(m, n)$ is defined as follows:
>
> $$
> A(m, n) = \begin{cases} n + 1\quad\quad &\text{if }m = 0 \\ A(m - 1, 1)\quad\quad &\text{if }m > 0\text{ and }n = 0 \\ A(m - 1, A(m, n - 1))\quad\quad &\text{if }m > 0\text{ and }n > 0 \end{cases}
> $$
> 
> For example $A(1, 0) = 2$, $A(2, 2) = 7$ and $A(3, 4) = 125$.
> 
> Find $\displaystyle\sum_{n=0}^6 A(n, n)$ and give your answer mod $14^8$.

* * *

The Ackermann function is notorious for being extremely fast growing. The first 4 terms of the sum ($A(0, 0)$ to $A(3, 3)$) are 1, 3, 7, and 61. However, $A(4, 4) = 2^{2^{2^{65536}}} - 3$, an absolutely monstrous number that is practically impossible to write down. It is completely infeasible to even attempt to calculate this number. However, because we need $\mod 14^8$, there is some hope. We will take this problem step-by-step, and build up to a solution that is manageable, and even doable by hand. The crux of the problem involves reducing the massive power of two in the values for $n\geq 4$.
## Background
### Tetration
We will start with $A(4, 4)$. The Wikipedia page for the [Ackermann function](https://en.wikipedia.org/wiki/Ackermann_function#Table_of_values) lists $A(4, 4) = 2 \uparrow\uparrow 7 - 3$. The double up-arrow notation is known as **Knuth arrow notation**, and in this case, it is **tetration.** Just like repeated addition is multiplication, and repeated multiplication is exponentiation, repeated exponentiation is tetration. We essentially create a "power-tower" of sorts, and $a\uparrow\uparrow b$ means to create a power-tower of $a$'s repeated $b$ times. When evaluating a power-tower directly, we always start from the top and work our down. For example, $2\uparrow\uparrow 4 = 2^{2^{2^2}} = 2^{2^4} = 2^{16} = 65536$. Even with relatively small values $a$ and $b$ can create absolutely massive numbers. 

Looking ahead, if you add an additional arrow to the notation, it extends to pentation, hexation, etc. It represents how many times to repeat the lower-order exponentiation. For example, $2\uparrow\uparrow\uparrow 3 = 2\uparrow\uparrow (2\uparrow\uparrow 2)$.  If $A(4, 4)$ is massive, then $A(5, 5)$ and $A(6, 6)$ are mind-bogglingly large, because those values grow according to pentation and hexation. We will come back to how we calculate this later.

### Carmichael function
Switching gears to something seemingly unrelated, let's talk about the [Carmichael function](https://en.wikipedia.org/wiki/Carmichael_function). We have witnessed the Euler totient function $\phi(n)$ (the number of numbers less than $n$ that are coprime to $n$) in past problems, and the Carmichael function $\lambda(n)$ is very closely related. Put simply, it is the smallest positive integer $m$ such that 

$$
a^m \equiv 1 \mod n
$$
is true for all $a$ coprime to $n$. Due to the modulus, it is enough to only check $a < n$. For example, $\lambda(8) = 2$, because all integers that are coprime to 8 (the odd numbers), are $1\mod 8$ when squared i.e. $1^2\equiv 3^2\equiv 5^2\equiv 7^2\equiv 1\mod 8$ (try it out!). An important fact is that  $\mathbf{\lambda(n)}$ **divides** $\mathbf{\phi(n)}$. Additionally, the Carmichael function can also be defined recursively using the totient function.

$$
\lambda(n) = 
\begin{cases}
    \phi(n) &\text{if }n\text{ is } 1, 2, 4,\text{ or an odd prime power},
    \\
    \frac{1}{2}\phi(n) &\text{if }n=2^r, r\geq 3,
    \\
    LCM\big( \lambda(n_1), \lambda(n_2), \dots, \lambda(n_k) \big) &\text{if }n = n_1n_2\dots n_k\text{ where } n_1, n_2, \dots, n_k\text{ are powers of distinct primes.}
\end{cases}
$$
This means we can use our existing solutions to calculate this function. Additionally, if we know the prime factorization of $n$, it is fairly easy to calculate $\lambda(n)$.

### Exponential Cycle
This is the most important fact and will allow us to solve this problem relatively easily. Let's say we are calculating $a^k\mod n$ for higher and higher values of $k$. At some point, the **sequence of results will start cycling.** In particular, the cycle length is related to $\lambda(n)$. If we have the prime factorization of $n=p_1^{r_1}p_2^{r_2}\dots p_k^{r_k}$, then for all $r\geq max(r_i)$,

$$
a^r \equiv a^{\lambda(n) + r}\mod n
$$
The one tricky bit is that $r$ needs to be at least the maximum exponent in the prime factorization. However, because the smallest prime factor is 2, we can increase this lower bound to  $r\geq \lfloor\log_2(n)\rfloor$, regardless if $n$ is divisible by 2 or not.
### Extremely Large Mods with the CRT
One last item we need to solve this problem is the question of how to solve $a^b\mod n$, where $b$ is an extremely large number. There is a way to use the binary form of $b$ and use successive multiplications, but we can also use the Chinese Remainder Theorem (CRT) to efficiently calculate the solution. The true CRT involves an arbitrary number of variables, but for the purposes of this problem, we will only need 2. The CRT basically states that a **unique** solution (up to mod) exists to the system

$$
\begin{cases}
    x &\equiv a_1\mod n_1 \\
    x &\equiv a_2\mod n_2 \\
\end{cases}
$$
where $n_1$ and $n_2$ are all coprime. To actually find $x$, we employ [Bézout's identity](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity), and find two integers $m_1$ and $m_2$ such that $m_1n_1+m_2n_2=1$. Then, a solution to the system is $x=a_1m_2n_2+a_2m_1n_1$. However, this will not be the smallest positive solution in most cases, and so we would need to add or subtract multiples of $n$ to get an equivalent answer.
#### Finding $m_1$ and $m_2$
To find both of these coefficients, we use the [extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm). The basic algorithm simply computes $gcd(n_1, n_2)$, but with the extended version, we find the coefficients such that $m_1n_1+m_2n_2=gcd(n_1, n_2)$. However, because $n_1$ and $n_2$ are coprime, their GCD is 1. In each step of the basic algorithm, the larger number is repeatedly replaced by the remainder when the larger is divided by the smaller, until one of the numbers is 0. The answer is the other nonzero number. The extended algorithm adds two extra parameters to keep track of, based on the highest multiple. Please refer to the Wikipedia page as it has a very clear worked out example. 

In our case, we will ensure that $n_1>n_2$ to ensure computations are easier to navigate.
## Solution
You might have noticed that many of the background sections mainly talk about computing basic exponentiation rather than tetration. However, the $2\uparrow\uparrow 7$ tetration is essentially exponentiation $2^C$, but $C$ is an absolutely monstrous number. Using the properties of the Carmichael theorem and its exponential cycle fact, we will repeatedly reduce $C$ to a value which can be easily stored, and then we will use the CRT to calculate the actual value.

Stating again, we have $a^r \equiv a^{\lambda(n) + r}\mod n$ where $r\geq \log_2(n)$.  As a direct example, $\lambda(100)=40$, and $\log_2(100) = 6.64$, so we can say that $2^7\mod 100 \equiv 2^{47}\mod 100 = 28$. Note that $a$ can be any value. However, **we can also work in the other direction.** If we want to know $2^{47}\mod 100$ we can subtract 40 off the exponent. **If we want to make $r$ as small as possible, we can take the modulus of $\lambda(n)$ itself. This leads to $r<\lambda(n)$ which can drastically make it easier to calculate.**
For example, let's say $r$ is incredibly big $\rightarrow 2^{4453}$. This number is very difficult to calculate on its own. However, $4453\equiv 13\mod 40$. Therefore,

$$
\begin{aligned}
2^{4453}\mod 100 &\equiv 2^{4453\mod\lambda(100)}\mod 100
\\ &\equiv 2^{4453\mod 40}\mod 100
\\ &\equiv 2^{13}\mod 100
\\ &\equiv 8192\mod 100
\\ &\equiv \boxed{92}
\end{aligned}
$$

At no point did I need to know $2^{4453}$ actually evaluates to. This will be the crux of how we can solve this problem without knowing what any of the extremely large powers evaluate to.
### Expansion to Tetration
How do we expand the above example to tetration? With tetration, the only difference is that instead of $4453$, we have a power tower. However, we can use the same mechanics, propagating smaller and smaller mods up the tower, and when we reach a point that is calculable by hand, we can propagate back downwards again. Additionally, $\lambda(\lambda(\dots(n)))$ will eventually reach 1. The only thing we need to keep in mind is that the powers need to be larger than $\log_2(m)$, where $m$ are the successive modulus on up. Here's an example with the tetration $2\uparrow\uparrow 6 = 2^{2^{2^{2^{2^2}}}}$ and $n=300$. The smallest tetration value which is "normal" is $2\uparrow\uparrow 4 = 65536$. This is where we'll stop. Going up through the power tower, we have,

$$
\begin{aligned}
    2\uparrow\uparrow 6\mod 300 &\equiv 2^{2\uparrow\uparrow 5 \mod\lambda(300)}\mod 300 \Rightarrow \lambda(300) = LCM(\lambda(2^2), \lambda(3), \lambda(5^2)) = LCM(2, 2, 20) = 20
    \\
    2\uparrow\uparrow 5\mod 20 &\equiv 2^{2\uparrow\uparrow 4 \mod\lambda(20)}\mod 20
    \\
    2\uparrow\uparrow 4\mod 4 &\equiv 65536\mod 4 \equiv \boxed{0}
\end{aligned}
$$
Now, when going back down the tower, we need to mindful of the exponent restriction $r\geq\log_2(n)$.

$$
\begin{aligned}
    2\uparrow\uparrow 5\mod 20 &\equiv 2^{2\uparrow\uparrow 4\mod \lambda(20)\,\,\, (\text{minimum }4)}\mod 20\,\,\Leftarrow \log_2(20) = 4
    \\
    &\equiv 2^{0 (\text{minimum }4)}\mod 20
    \\
    &\equiv 2^4 \mod 20 \Leftarrow 0\mod 4\equiv 4\mod 4 \text{ so we replace with }4
    \\
    &\equiv 16\mod 20 = \boxed{16}
    \\ \\
    2\uparrow\uparrow 6\mod 300 &\equiv 2^{2\uparrow\uparrow 20\mod\lambda(300)\,\,\, (\text{minimum }8)}\mod 100\,\,\,\Leftarrow \log_2(300) = 8
    \\
    &\equiv 2^{16 (\text{minimum }8)}\mod 300 \Leftarrow 16>8\text{ so we take }16
    \\
    &\equiv 2^{16}\mod 300 = \boxed{136}
\end{aligned}
$$

This is apparent in the 3rd step of evaluating $2\uparrow\uparrow 5$. Our previous answer was $0$, but we needed a minimum of 4 in the exponent in order to use the exponential cycle. The smallest integer greater than 0 that is equivalent to $0\mod 4$ (this 4 comes from $2\uparrow\uparrow 4\mod 4$) is 4 itself. The next evaluation doesn't need this check, since $16>8$. And that is it! We evaluated a monstrous number modulus another.

### Other cases
There are still some loose ends to tie up. In the steps shown above, the raw exponent values will be less than $\lambda(n)$. However, when dealing with large modulus (such as this problem), it is also possible that $\lambda(n)$ is itself large. Instead of $2^{16}\mod 300$ in the last step, what if we had $2^{48278}\mod14^8$? Here is where we can use the Chinese Remainder Theorem and Bezout's Identity to calculate this easily.
### Pentation and Hexation
I have showed methods for tetration, but how do we handle cases with more arrows e.g. $A(5, 5) = 2\uparrow\uparrow\uparrow 7 - 3$. Pentation is really just repeated tetration. This is equal to $2\uparrow\uparrow B$, where $B$ is an absolutely humongous number. If we don't know $B$, how will our method work? Remember how we repeatedly take $\lambda(n)$ over and over again? $\lambda(n)$ will never be bigger than $n$, and so eventually, this will reduce to 1. This means after a certain exponent $k$, $2\uparrow\uparrow k\mod n$ will remain **constant**. This means both $A(5,5)=A(6,6)$ and that we can use a **much, much smaller** value for $k$ then the true values, and repeat the same process until the Carmichael numbers have reduced to 1.
## Solution
Now we finally have all the pieces to solve this problem. $A(0,0)$ through $A(3,3)$ can be solved using the regular recurrence relation. After that we use the Carmichael numbers, which can be calculated from the Euler totient values using simple rules for $p^k$, where $p$ is prime. We calculate $A(4, 4)$ with this method. For both $A(5, 5)$ and $A(6, 6)$, we can simply substitute a much smaller value, since all results above a certain value will evaluate to the same result. Thus, it is enough to calculate $2\uparrow\uparrow 200 \mod 14^8$ for $A(5, 5)$ and $A(6, 6)$. The function to calculate this works nicely in recursive form. Our base cases are when either the power is 1, or $n$ is 1. Otherwise we calculate $\lambda(n)$ and its prime factorization (which can be read directly from $LCM$), and pass it in the sub-call. We also use our existing functions to calculate a "large modulus" and the prime factorization, as well as the original Ackermann recursive function definition.

```python
import time
import primesieve
import numpy as np
from collections import defaultdict

# Function to find prime factorization
# of number, given list of primes...
def prime_fact(n, primes):
    if n == 1:
        return {}
    factors = []
    j = 0  # pointer for prime...
    while n != 1:
        # Find smallest prime that divides n...
        while n % primes[j] != 0:
            j += 1
        # Divide the prime as many times as it takes...
        count = 0
        while n % primes[j] == 0:
            n //= primes[j]
            count += 1
        # Add the pair...and increment j...
        factors.append((primes[j], count))
        j += 1
    return dict(factors)


def carmichael(n, primes, factorization=None):
    if n == 1:
        return 1
    if factorization is None:
        factorization = prime_fact(n, primes)  # First, prime factorize it...
    # Carmichael is the LCM of each of the Carmichaels...
    # We need prime factorization for LCMs.
    # When given the prime factorizations of each individual component,
    # the LCM is just the max power of each prime we see.
    # Go through each distinct prime power and find the factorization
    # of each. But since the Totient function is simple to calculate
    # for prime powers phi(p^r) = p^(r-1)(p-1), we can build up the LCM...
    # The consequence of this is that we also calculate the factorization of lambda(n) itself.
    lcm_dict = {}
    for prime, power in factorization.items():
        # Two cases, either we have (1, 2, 4, and odd prime power) or we have 2^r, with r >= 3
        if prime > 2 or (prime == 2 and power <= 2):
            # In this case, lambda(n) = phi(n) = p^(r-1)(p-1)
            # We also need to calculate the factorization of p - 1
            p_minus_1_fact = prime_fact(prime - 1, primes)
            car_component_dict = {prime: power - 1} | p_minus_1_fact
        else:
            # Otherwise, we have 2^r with r >= 3, and lambda(n) = 1/2 phi(n).
            # But since n is a power of 2, phi(2^r) = 2^(r-1) ==> lambda(2^r) = 2^(r-2)
            car_component_dict = {2: power - 2}
        # Merge with the LCM dict.
        # Compare the factorization in the component with the existing, and take the maximum
        # power in case a prime is found in both.
        # Go through the component and remove any prime powers that are LESS THAN than what's in the LCM,
        # and then combine with the pipe syntax, which takes the latter if keys are found in both...
        # Dictionaries are immutable, which means we need to create a new dict in memory...
        car_component_dict = {prime: power for prime, power in car_component_dict.items()
                              if (prime not in lcm_dict) or (prime in lcm_dict and power > lcm_dict[prime])}
        # fact_primes = car_component_dict.keys()
        # for prime in fact_primes:
        #     power = car_component_dict[prime]
        #     if prime in lcm_dict and power <= lcm_dict[prime]:
        #         del car_component_dict[prime]
        lcm_dict = lcm_dict | car_component_dict
    # Remove any 0 powers...
    lcm_dict = {k: v for k, v in lcm_dict.items() if v != 0}
    # Evaluate the number given the LCM factorization
    res = 1
    for prime, power in lcm_dict.items():
        res *= prime ** power
    return res, lcm_dict


# Calculates a^b mod c for large a^b
def large_mod(a, b, c):
    if c == 1 or c == a:
        return 0
    binaryB = bin(b)[2:][::-1]
    answer = 1
    if binaryB[0] == '1':
        answer = a
    currPowerMod = a
    for power, binDigit in enumerate(binaryB[1:]):
        currPowerMod = (currPowerMod * currPowerMod) % c
        if binDigit == '1':
            answer = (answer * currPowerMod) % c
    return answer


def ackermann(m, n):
    if m == 0:
        return n + 1
    if m > 0:
        if n == 0:
            return ackermann(m - 1, 1)
        return ackermann(m - 1, ackermann(m, n - 1))


def double_arrow_mod(A, T, n, prime_fact_n, min_required, primes):
    # This will be a recursive function. The modulus will keep reducing
    # according to the Carmichael function. Eventually, it will reach one.
    # If we are modding 1, then the answer is always 0 (every number is divisible by 1).
    # However, we need to take into account the min_required and increase accordingly...
    # So it's actually min_required instead.
    if n == 1:
        return min_required
    # It's also possible T is 1 before n is.
    if T == 1:
        return A % n
    min_required_next = max(prime_fact_n.values())
    # Calculate lambda(n) and its factorization, passing in the factorization of n
    car_num, car_powers = carmichael(n, primes, factorization=prime_fact_n)
    # Find the power that should be much smaller
    new_power = double_arrow_mod(A, T - 1, car_num, car_powers, min_required_next, primes)
    # Once we have the power, we use large_mod to calculate the answer.
    # Make sure to update it in case it is < min_required!
    result = large_mod(A, new_power, n)
    while result < min_required:
        result += n
    return result


start = time.perf_counter()

mod = 14 ** 8
primes = primesieve.primes(mod ** 0.5)
prime_fact = prime_fact(mod, primes)

twoDAseven = double_arrow_mod(2, 7, mod, prime_fact, 0, primes)
twoDAmonster = double_arrow_mod(2, 200, mod, prime_fact, 0, primes)
ackSum = sum(ackermann(i, i) for i in range(4))
ackSum += (twoDAseven - 3) + 2 * (twoDAmonster - 3)
print(ackSum % mod)

end = time.perf_counter()
print(end - start, 'seconds.')
```
Running this, we get

```text
1098988351
0.0005780000064987689 seconds.
```

Therefore, the answer to the problem with absurdly large raw numbers is **1098988351**. Notice how quick our program ran in the end!
