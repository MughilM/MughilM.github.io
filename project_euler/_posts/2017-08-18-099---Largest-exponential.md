---
layout: post
title: "#99 - Largest exponential"
date: 2017-08-18 01:40
number: 99
tags: [10_diff]
---
> Comparing the two numbres written in index form like $2^{11}$ and $3^7$ is not difficult, as any calculator would confirm that $2^{11}=2048<3^7=2187$.
> 
> However, confirming that $632382^{518061}>519432^{525806}$ would be much more difficult, as both numbers contain over three million digits.
> 
> Using [base_exp.txt](https://projecteuler.net/project/resources/p099_base_exp.txt) (right click and 'Save Link/Target As...'), a 22K text file containing one thousand lines with a base/exponent pair on each line, determine which line number has the greatest numerical value.
> 
> The first two lines in the file represent the numbers in the example given above.
> {:.note}
{:.lead}
* * *

Even calculating all these large numbers in Python is not that efficient. We can take the **logarithm** of both sides, and retain the monotonicity of the exponentials. So if beforehand, $a^b > c^d$, then $\log a^b > \log c^d$. We can simplify this further with $b\log a > d\log c$, which will be the condition we'll check.
```python
# file: "problem099.py"
exponents = np.genfromtxt(
    'p099_base_exp.txt', delimiter=',', dtype=np.int64)
logs = exponents[:, 1] * np.log10(exponents[:, 0])
# Now find the maximum argument and print its line
index = np.argmax(logs)
print(index + 1)
print(exponents[index].tolist())
```
Running this short code results in an output,
```
709
[895447 504922]
0.011759600000000037 seconds.
```
Thus, $895447^{504922}$ is the largest number in our list, and it happens at line **709**.