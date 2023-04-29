---
layout: post
title: "#191 - Prize Strings"
date: 2019-07-13 19:18
number: 191
tags: [35_diff]
---
> A particular school offers cash rewards to children with good attendance and punctuality. If they are absent for three consecutive days or late on more than one occasion then they forfeit their prize.
>
> During the $n$-day period a trinary string is formed for each child consisting of L's (late), O's (on time), and A's (absent).
>
> Although there are eight-one trinary strings for a 4-day period that can be formed, exactly forty-three strings would lead to a prize:
>
> <pre>
>     OOOO OOOA OOOL OOAO OOAA OOAL OOLO OOLA OAOO OAOA
> OAOL OAAO OAAL OALO OALA OLOO OLOA OLAO OLAA AOOO
> AOOA AOOL AOAO AOAA AOAL AOLO AOLA AAOO AAOA AAOL
> AALO AALA ALOO ALOA ALAO ALAA LOOO LOOA LOAO LOAA
> LAOO LAOA LAAO
> </pre>
>
> How many "prize" strings exist over a 30-day period?
{:.lead}
* * *

We want all strings of length $n$ where we **do not** have 3 consecutive A's **nor** more than one L. It's easier to count the complement, and subtract from the total. So we'll find the number of strings with less than 2 L's and 3 consecutive A's.
## Less than 2 L's
We break this up into two cases: strings without any L's, and strings with exactly one L:
* **Zero L's**: Our only option is "O" or "A" for each letter. With $n$ letters, there are $2^n$ such strings.
* **One L**: We can place the "L" in any one of $n$ spots. The other $n-1$ letters have to be either "O" or "A". Thus, there are $n2^{n-1}$ such strings.

Together, we have $2^n+n2^{n-1} = 2^{n-1}(n+2)$ strings with less than two Ls.
## Three consecutive A's AND less than 2 L's
This is more involved, and like before, we break this into cases. Let $f(n)$ be the number of strings with no L's and 3 consecutive A's, and $g(n)$ be the number of strings with one L and 3 consecutive A's.
### Finding $f(n)$
Assume we have one such $n$-character string. Then it can be exactly one of the following:
* **Ends with "AAA"**: The first $n-3$ characters can be any of "O" or "A". There are $2^{n-3}$ of these.
* **Ends with "O"**: The first $n-1$ characters must contain 3 consecutive A's and no L's, of which there are $f(n-1)$.
* **Ends with "OA"**: The first $n-2$ characters must contain 3 consecutive A's and no L's, of which there are $f(n-2)$.
* **Ends with "OAA"**: This set contains $f(n-3)$ strings.

Therefore, our formula for the number of strings with 3 consecutive A's and no L's is

$$
f(n)=2^{n-3}+f(n-1)+f(n-2)+f(n-3)=\boxed{2^{n-3}+\sum_{i=1}^3f(n-i)}
$$

### Finding $g(n)$
We can still break these into cases:
* **Ends with "AAA"**: The first $n-3$ characters can be any of "O" or "A", *in addition to the fact that we need one L*. There are $n-3$ spots to place the L, and the other $n-4$ can be any of the other two. In total, there are $(n-3)2^{n-4}$.
* **Ends with "O"**: The first $n-1$ must be one of $g(n-1)$ strings. You can immediately see what the next two cases amount to.
* **Ends with "OA"**: $g(n-2)$.
* **Ends with "OAA"**: $g(n-3)$.
* **Ends with "L"**: This is one of the extra cases we must consider. In this case, the first $n-1$ characters *must contain 3 consecutive A's*. How many strings is that? Well that's just the $f(n-1)$ we calculated in the previous section!
* **Ends with "LA"**: The first $n-2$ must have "AAA", which count for $f(n-2)$ strings.
* **Ends with "LAA"**: $f(n-3)$.

Adding up all the cases, we get

$$
\begin{aligned}
	g(n) &= (n-3)2^{n-4}+\sum_{i=1}^3(g(n-i)+f(n-i))
	\\ &=
	(n-3)2^{n-4}+\sum_{i=1}^3g(n-i)+\sum_{i=1}^3 f(n-i)
	\\ &=
	(n-3)2^{n-4}+\sum_{i=1}^3 g(n-i)+ f(n)-2^{n-3}
	\\ &=
	\boxed{2^{n-4}(n-5)+\sum_{i=1}^3 g(n-i)}
\end{aligned}
$$
## Final amount
We take the number of strings which have zero or one L, and subtract off those which contain 3 consecutive A's. In all, this is

$$
a(n)=2^{n-1}(n+2)-f(n)-g(n)
$$
Since $f(n)$ and $g(n)$ are recursive, we calculate all of these beforehand, and a single line to calculate the final amount is needed.
```python
# file: "problem191.py"
limit = 30
# Make the fn and gn arrays up until limit...
f = [0] * (limit + 1)
g = [0] * (limit + 1)
# fn = 2^(n-3) + (3 previous terms)
# gn = 2^(n-4)*(n-5) + (3 previous terms) + fn
for n in range(3, limit + 1):
    f[n] = 2 ** (n - 3) + sum(f[n - 3:n])
    g[n] = 2 ** (n - 4) * (n - 5) + sum(g[n - 3:n]) + f[n]

# Answer is 2^(n-1) * (n+2) - fn - gn
print(int(2 ** (limit - 1) * (limit + 2) - f[limit] - g[limit]))
```
Running this short code, we get
```
1918080160
7.449999999997736e-05 seconds.
```
Therefore, there are **1918080160** such prize strinsg for 30 days.