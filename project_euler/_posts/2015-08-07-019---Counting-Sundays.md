---
layout: post
title: "#19 - Counting Sundays"
date: 2015-08-07 20:49
number: 19
tags: [05_diff]
---
> You are given the following information, but you may prefer to do some research for yourself.
> 
> * 1 Jan 1990 was a Monday.
> * Thirty days has September.
> April, June and November.
> All the rest have thirty-one,
> Saving February alone,
> Which has twenty-eight, rain or shine.
> And on leap years, twenty-nine.
> * A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
> 
> How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
{:.lead}
* * *

This problem is slightly made easier by the fact that we only have to keep track of the firsts of the months. There are 7 days per week, which rotate on a cycle. We can use modular arithmetic to handle continuous rotations. The numbers 0-6 will be used to denote Sunday through Saturday. As stated in the problem, January 1st 1990 was a Monday. January has 31 days as well, so

$$
31 \equiv 3\mod 7
$$
means that January has 4 full weeks, plus 3 extra days. Therefore, add 3 days to Monday, and we can conclude that February 1st 1990 was a Thursday. We do this each month starting from January 1st 1901 until we hit the end of the century. The year 1900 was **not** a leap year, so it had 365 days. Thus, January 1st 1901 was a Tuesday (since $365\equiv 1\mod 7$). Any time the number is 0, that represents a Sunday and we have to count it. We hardcode the number of days per month, and we also need to keep track of leap years, which is straightforward.
```python
# file: "problem019.py"
# Days will be numbered 0, 1, 2, ... -> Sunday, Monday, Tuesday, ...
# Months will be numbered 0-11.
# Given that 1 Jan 1990 is Monday
firstDay = 1
# We need to start counting from Jan 1 1901 though.
# 1900 was NOT a leap year, so there were 365 days...
firstDay += 365 % 7
# Array to hold number of dayskfor each month from Jan to Dec
monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
numOfSundays = 0
for year in range(1901, 2001):
    for month in range(12):
        # Check if it's a sunday...
        if firstDay == 0:
            numOfSundays += 1
        # First check if it's a leap year.
        # We need to mod afterwards as well
        # Because there is a chance the addition will take it over 7...
        # Remember 0-indexing, so month == 1 means it's the second month.
        if (month == 1) and ((year % 100 != 0 and year % 4 == 0) or (year % 400 == 0)):
            firstDay = (firstDay + 29 % 7) % 7
        else:
            firstDay = (firstDay + monthDays[month] % 7) % 7

print(numOfSundays)
```
Running the code gives an output of,
```
171
0.0008150123456790123 seconds.
```
Therefore, there are **171** Sundays that fall on the first of the month from January 1st 1901 to December 31st 2000.