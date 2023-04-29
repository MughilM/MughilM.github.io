---
layout: post
title: "#32 - Pandigital products"
date: 2016-05-09 16:53
number: 32
tags: [05_diff]
---
> We shall say that an $n$-digit number is pandigital if it makes use of all the digits 1 to $n$ exactly once; for example, the 5-digit number, 15234, is 1 through 5 pandigital.
> 
> The product 7254 is unusual, as the identity, $39\times 186=7254$, containing multiplicand, multiplier, and product is 1 through 9 pandigital.
> 
> Find the sum of all products whose multiplicand/multiplier/product identity can be written as a 1 through 9 pandigital.
> 
> Some products can be obtained in more than one way so be sure to only include it once in your sum.
> {:.note}
{:.lead}
* * *

We can use the `itertools.permutations` function to loop through all 9-digit arrangements of 1-9. Next, we can place the multiplication and equals signs at valid positions. The former can be placed anywhere between the 1st and 8th digit (to accomodate the equals sign afterwards). We would test all arrangements to see if a valid equation is made.

For example, in the problem above, we would start from $3\times 9=1867254$ and end with $3918672\times 5=4$. However, as we loop through on places to put the equals sign, the product decreases as we move left to right. Eventually it will be less than the "correct" product (what the equation _should_ equal). Then we can stop and move the multiplication sign over one. In the same example, the last product we check is $3\times 918=67254$, because the next product $3\times 9186>7254$. In this manner, we skip a good number of possibilities.
```python
# file: "problem032.py"
products = set()
for perm in permutations('123456789'):
    # We place a multiplication and equals
    # sign at all possible places here.
    # Once the product goes above the RHS,
    # we can stop and test putting the
    # multiplication sign in the next spot.
    # The multiplication symbol can be placed
    # after the 1st digit, or before the
    # second to last digit (because we need equals)
    for i in range(1, len(perm) - 2):
        # Now we place an equals sign anywhere
        # from at least one digit after
        # the multiplicand to one before the result.
        # Once the product becomes larger than RHS,
        # we stop, because the RHS will only get smaller.
        for j in range(i + 1, len(perm) - 1):
            mCand = int(''.join(perm[:i]))
            mPlier = int(''.join(perm[i:j]))
            product = mCand * mPlier
            rhs = int(''.join(perm[j:]))
            if product > rhs:
                break
            elif product == rhs:
                products.add(product)

print(sum(products))
```
We add the products to a `set()` to avoid duplicates. Running the double for loop gets an output of
```
45228
9.040594220951176 seconds.
```
Therefore, the sum of all products is **45228**. It might be possible to cut the time in half, as multiplication is commutative.