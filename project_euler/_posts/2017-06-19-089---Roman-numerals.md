---
layout: post
title: "#89 - Roman numerals"
date: 2017-06-19 14:07
number: 89
tags: [20_diff]
---
> For a number written in Roman numerals to be considered valid there are basic rules which must be followed. Even though the rules allow some numbers to be expressed in more than one way there is always a "best" way of writing a particular number.
> 
> For example, it would appear that there are at least six ways of writing the number sixteen:
> 
> <pre style="text-align:left">
> IIIIIIIIIIIIIIII
> VIIIIIIIIIII
> VVIIIIII
> XIIIIII
> VVVI
> XVI
> </pre>
> 
> However, according to the rules only `XIIIIII` and `XVI` are valid, and the last example is considered to be the most efficient, as it uses the least number of numerals.
> 
> The 11K text file, [roman.txt](https://projecteuler.net/project/resources/p089_roman.txt) (right click and 'Save Link/Target As...'), contains one thousand numbers written in valid, but not necessarily minimal, Roman numerals; see [About... Roman Numerals](https://projecteuler.net/about=roman_numerals) for the definitive rules for this problem.
> 
> Find the number of characters saved by writing each of these in their minimal form.
> 
> Note: You can assume that all the Roman numerals in the file contain no more than four consecutive identical units.
> {:.note}
{:.lead}
* * *

Barring some special edge cases (please see the link) the rule for creating Roman numerals boils down to: **From left to right, the numeral must NOT increase, UNLESS it is following the rule of subtraction.** 

We need functions that convert back and forth. Converting roman numerals to numbers is straightforward; just look at each pair of numerals and add them to a running total (accounting for subtraction rules when necessary). Going the other direction is a bit more tricky. Looking at the rules, we can deduce that there are really **13 possible denominations**:
* 1000 - M
* 900 - CM
* 500 - D
* 400 - CD
* 100 - C
* 90 - XC
* 50 - L
* 40 - XL
* 10 - X
* 9 - IX
* 5 - V
* 4 - IV
* 1 - I

We need to continuously check what the biggest denomination less than the given number is. We then subtract off that value, and keep repeating until the number is 0.
```python
# file: "problem089.py"
def romanToNum(romanStr):
    romanVal = {'M':1000, 'D':500, 'C':100,
                'L':50, 'X':10, 'V':5, 'I':1}
    value = 0
    i = 0
    while i < len(romanStr):
        r = romanStr[i]
        if i < len(romanStr) - 1 and romanVal[r] < romanVal[romanStr[i+1]]:
            value += romanVal[romanStr[i+1]] - romanVal[r]
            i += 2
        else:
            value += romanVal[r]
            i += 1
    return value
def numToRoman(n):
    roman = ''
    # The values that can be encoded are
    # 1000, 900, 500, 400, 100, 90, 50, 40,
    # 10, 9, 5, 4, and 1 with one or two numerals
    # So count down
    values = [1000, 900, 500, 400,
              100, 90, 50, 40, 10,
              9, 5, 4, 1]
    codings = ['M', 'CM', 'D', 'CD',
               'C', 'XC', 'L', 'XL',
               'X', 'IX', 'V', 'IV', 'I']
    mappings = {value: code for value, code in zip(values, codings)}
    while n > 0:
        # Find the greatest denomination that
        # is less than n...
        i = 0
        while values[i] > n:
            i += 1
        val = values[i]
        # Get the mapping, and subtract it off...
        roman += mappings[val]
        n -= val
    return roman

lettersSaved = 0
with open('p089_roman.txt') as f:
    romans = f.read().splitlines()
for roman in romans:
    lettersSaved += len(roman) - len(numToRoman(romanToNum(roman)))

print(lettersSaved)
```
Running this code gets us an output of,
```
743
0.022997900000000016 seconds.
```
Thus, we save **743** letters if we write each roman numeral in its minimal form.