---
layout: post
title: "#79 - Passcode derivation"
date: 2017-06-19 11:19
number: 79
tags: [05_diff]
---
> A common security method used for online banking is to ask the user for three random characters from a passcode. For example, if the passcode was 531278, they may ask for the 2nd, 3rd, and 5th characters; the expected reply would be: 317.
> 
> The text file, [keylog.txt](https://projecteuler.net/project/resources/p079_keylog.txt), contains fifty successful login attempts.
> 
> Given that the three characters are always asked for in order, analyze the file so as to determine the shortest possible secret passcode of unknown length.
{:.lead}
* * *

Looking through the successful login attempts, two things jump out:
* Every code starts with a 7
* Every code ends with a 0

This means that the true passcode starts and ends with a 7 and 0 respectively.

Some assumptions before we start searching:
* Each digit appears only once in the full passcode, since we are looking for the shortest possible passcode, and no digit appears twice within each login attempt.
* No 4s or 5s are in the passcode, since no login attempt had one.

We will store the possibilites of what numbers can come before and after a certain digit by using a dictionary which maps each number to a 2-tuple.

In order to build the full passcode, we see which digit has **no** possible digits that come before it. After we've selected it, we remove that number from all other instances. We repeat process, until we have exhausted through everything. The ending string is the full passcode.
```python
# file: "problem79.py"
# Set a bunch of rules, like digit x must come before y.
rules = {str(digit): (set(), set()) for digit in range(10) if digit != 4 and digit != 5}

with open('./p079_keylog.txt', 'r') as f:
    keylogs = f.read().splitlines()

for log in keylogs:
    for i in range(len(log)):
        if i == 0:
            rules[log[i]][1].add(log[i+1])
        elif i == len(log) - 1:
            rules[log[i]][0].add(log[i-1])
        else:
            rules[log[i]][0].add(log[i-1])
            rules[log[i]][1].add(log[i+1])
# Now we see which digit has nothing
# come before it. Once we've found it,
# we remove that from all other lists,
# and perform the search again...
password = ''
while len(password) < 8:
    nextDigit = None
    for key, values in rules.items():
        if len(values[0]) == 0:
            nextDigit = key
            break
    password += nextDigit
    del rules[nextDigit]
    # Remove it from the rest...
    for key, values in rules.items():
        if nextDigit in values[0]:
            values[0].remove(nextDigit)

print('The password is', password)
```
Running the above code, we get
```
The password is 73162890
0.0008719999999999839 seconds.
```
Thus, our full passcode is **73162890**.