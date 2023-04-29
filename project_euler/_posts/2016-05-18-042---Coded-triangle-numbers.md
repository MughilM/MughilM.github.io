---
layout: post
title: "#42 - Coded triangle numbers"
date: 2016-05-18 20:31
number: 42
tags: [05_diff]
---
> The $n^{\text{th}}$ term of the sequence of triangle numbers is given by, $t_n=\frac{1}{2}n(n+1)$; so the first ten triangle numbers are:
> 
> $$
> 1,3,6,10,15,21,28,36,45,55,\dots
> $$
> 
> By converting each letter in a word to a number corresponding to its alphabetical position and adding these values we form a word value. For example, the word value for SKY is $19+11+25=55=t_{10}$. If the word value is a triangle number then we shall call the word a triangle word.
> 
> Using [words.txt](https://projecteuler.net/project/resources/p042_words.txt) (right click and 'Save Link/Target As...'), a 16K text file containing nearly two-thousand common English words, how many are triangle words?
{:.lead}
* * *

Let $w_{word}$ by the word value for $word$. We need to see whether $w_{word}$ is a triangle number. If it is, then by using the quadratic formula,

$$
\begin{aligned}
w_{word} &= \frac{1}{2}n(n+1) \\
n^2+n-2w_{word} = 0 \\
n &= \frac{-1 \pm \sqrt{1-4(1)(-2w_{word})}}{2}
\\ &=
\frac{\sqrt{8w_{word} + 1} - 1}{2}
\end{aligned}
$$

Therefore, for $w_{word}$ to be a triangle number, the last expression has to be an integer, which means $8w_{word}+1$ has to be a square, which will be our simple test.

We loop over all words, and check if the word value matches the condition. I've saved the file `words.txt` as `problem042.txt`. The words are in double quotes, so we'll need to remove them to get the pure words.
```python
# file: "problem042.py"
with open("problem042.txt") as f:
    words = f.readlines()
# split so it becomes 1D array
words = words[0].split(',')
# Now traverse through the words and remove
# the double quotes on the ends
for i in range(len(words)):
    words[i] = words[i].replace('"', '')

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
count = 0
for i in range(len(words)):
    num = 0
    for j in range(len(words[i])):
        num += letters.index(words[i][j:j+1]) + 1
    if (8 * num + 1) ** 0.5 == int((8 * num + 1) ** 0.5):
        count = count + 1

print(count)
```
The output is,
```
162
0.011115430442436482 seconds.
```
Thus, there are **162** triangle words in the file.