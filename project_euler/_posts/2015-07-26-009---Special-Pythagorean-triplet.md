---
layout: post
title: "#9 - Special Pythagorean triplet"
date: 2015-07-26 19:24
number: 9
tags: [05_diff]
---
> A Pythagorean triplet is a set of three natural numbers, $a < b < c$, for which,
>
> $$
> a^2 + b^2 = c^2
> $$
> 
> For example, $3^2 + 4^2 = 9 + 16 = 25 = 5^2$.
>
> There exists only one Pythagorean triplet for which $a + b + c = 1000$.
> Find the product $abc$.
{:.lead}
* * *

The brute force method would simply loop through all $a,b,c$ values from 1 to 999. However, we can use our given relationships to reduce the search space to just two variables. Note that 

$$
(a + b + c)^2 = a^2 + 2ab + b^2 + 2ac + 2bc + c^2
$$

From $a + b + c = 1000$, we have $c = 1000 - a - b$. Using these facts and the Pythagorean theorem, we can simplify the above expression:

$$
\begin{aligned}
	1000^2 &= 2ab + 2ac + 2bc + 2c^2
	\\
	\frac{1000^2}{2} &= ab + a(1000 - a - b) + b(1000 - a - b) + (1000 - a - b)^2
	\\
	\frac{1000^2}{2} &= ab + 1000a - a^2 - ab + 1000b - ab - b^2 - 1000^2 - 2(1000)(a+b) + (a+b)^2
	\\
	\frac{1000^2}{2} &= 1000a-a^2+1000b-ab-b^2-1000^2-2000a-2000b+a^2+2ab+b^2
	\\
	\frac{1000^2}{2} &= 1000^2-1000a-1000b+ab
	\\
	500 &= 1000-a-b+\frac{ab}{1000}
	\\
	a+b-500 &= \frac{ab}{1000}
\end{aligned}
$$

The last statement provides our condition to check for all pairs $a, b$. So we have reduced to a double for loop.
```python
# file: "problem009.py"
for b in range(1, 999):
    for a in range(1, b):
        if a + b - 500 == a * b / 1000:
            c = int((a ** 2 + b ** 2) ** 0.5)
            print('(a, b, c)', '=', a, b, c)
            print('abc', '=', a * b * c)
            end = time.perf_counter()
            print(end - start, 'seconds.')
            sys.exit(0)
```
Running the above code gives us,
```
(a, b, c) = 200 375 425
abc = 31875000
0.033183526281024125 seconds.
```
Therefore, our product is **31875000**.
