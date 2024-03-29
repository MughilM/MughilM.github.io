---
layout: post
title: "#62 - Cubic permutations"
date: 2018-05-27 15:36
number: 62
tags: [15_diff]
---
> The cube, $41063625 \left(345^3\right)$, can be permuted to produce two other cubes: $56623104 \left(384^3\right)$ and $66430125 \left(405^3\right)$. In fact, 41063625 is the smallest cube which has exactly three permutations of its digits which are also cube.
> 
> Find the smallest cube for which exactly five permutations of its digits are cube.
{:.lead}
* * *

Notice they didn't give us any digit bounds. We keep increasing the number of digits, until we get a set of 5 permutations. For each number of digits $k$, we generate all cubes that have $k$ digits. Next, we sort each cube on its digits. One cube is a permutation of another if its sorted set of digits is exactly the same. Therefore, the thing we are looking for is the same sorted digit string appearing 5 times. 

Doing `sorted(str())` will convert to a sorted string, and we can vectorize it so that it applies to every element in an array. The `np.unique` function can allow us to get the counts of each permutation.
```python
# file: "problem062.py"
nDigs = 2
foundCube = False
# The following function sorts the digits
# of a number
sortDigits = lambda x: "".join(sorted(str(x)))
# Vectorize so it can be applied to each
# element in one go.
vectSortDigs = np.vectorize(sortDigits)
while not foundCube:
    # Find the upper and lower bound cube for these digits.
    lower = int(10 ** ((nDigs - 1) / 3)) + 1
    upper = int(10 ** (nDigs / 3))
    # Form an array of numbers cubed
    nums = np.arange(lower, upper + 1, dtype=np.uint64) ** 3
    # Get the sorted list of digits for each number...
    perms = vectSortDigs(nums)
    # Get all unique permutations and their counts
    uniques, counts = np.unique(perms, return_counts=True)
    # We check here whether or not we got 5 perms
    # Grab the index of the 5 perm, if it's not there
    # it will throw an IndexError
    try:
        fivePerm = uniques[np.where(counts == 5)[0][0]]  # Will throw IndexError if it doesn't exist
        origNums = nums[np.where(perms == fivePerm)[0]]
        print('The five cubes are: ', origNums)
        print('The cube roots are: ', origNums ** (1/3))
        print(nDigs)
        foundCube = True
    except IndexError:
        pass  # Skip, we didn't find a 5 set with this many digits...
    nDigs += 1
```

Running the loop gets us an output of,

```
The five cubes are:  [127035954683 352045367981 373559126408 569310543872 589323567104]
The cube roots are:  [5027. 7061. 7202. 8288. 8384.]
12 digits
0.030961370484396847 seconds.
```
Therefore, $5027^3, 7061^3, 7202^3, 8288^3$ and $8384^3$ are all permutations of each other, with the smallest being **127035954683**.