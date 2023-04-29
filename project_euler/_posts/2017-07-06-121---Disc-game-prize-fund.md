---
layout: post
title: "#121 - Disc game prize fund"
date: 2017-07-06 10:16
number: 121
tags: [35_diff]
---
> A bag contains one red disc and one blue disc. In a game of chance a player takes a disc at random and its colour is noted. After each turn the disc is returned to the bag, an extra red disc is added, and another disc is taken at random.
>
> The player pays £1 to play and wins if they have taken more blue discs than red discs at the end of the game.
>
> If the game is played for four turns, the probability of a player winning is exactly 11/120, and so the maximum prize fund the banker should allocate for winning in this game would be £10 before they would expect to incur a loss. Note that any payout will be a whole number of pounds and also includes the original £1 paid to play the game, so in the example given the player actually wins £9.
>
> Find the maximum prize fund that should be allocated to a single game in which fifteen turns are played.
{:.lead}
* * *

The key takeaway is that since we are adding a red disc each turn, **the probability of drawing a blue disc decreases each turn**.

To find the maximum prize fund, we need to find the probability of winning game, or the probability of drawing 8 or more blue discs in the 15 turns. **If** the probability stayed constant each time, then this a binomial distribution. If the probability of drawing a blue disc was $p$ at each turn, then the probability of drawing 8 blue discs out of 15 tries is

$$
P(8\text{ blue discs}) = \binom{15}{8}p^8(1-p)^7
$$
The binomial coefficient arises because you can draw the 8 discs in different ways i.e. drawing them in your first 8 tries vs. drawing in last 8 tries. Since each turn is independent, we multiply $p$ 8 times, and $1-p$ 7 times.

However, $p$ is different depending on the turn in our case. Drawing a blue disc in your first 8 turns have a different probability than drawing in the last 8 turns. Thus, we loop through each of the $\binom{15}{8}$ possibilities and calculate the probability of drawing 8 blue discs. Each possibility is mutually exclusive, so we add the results in the end.

We then need to find the greatest amount of money that can be allocated to be still seen as a win from the casino's eyes, or a loser from the player's eyes. Since the player pays £1 to play, the max amount $M$ such that $MP<1 \Rightarrow M < \frac{1}{P}$, where $P$ is the probability of winning the game.

To loop through each blue disc progression, we use `itertools.combinations` with the included parameter being the number of blue discs we draw. Yielding the order allows for on the fly generation.
```python
# file: "problem121.py"
def kDiscsFromN(n, k):
    for blues in combinations(range(n), k):
        order = ['R'] * n
        for blue in blues:
            order[blue] = 'B'
        yield ''.join(order)

N = 15
bluesToWin = N // 2 + 1
# Calculate the win probability
winProb = 0
for blues in range(bluesToWin, N + 1):
    for progression in kDiscsFromN(N, blues):
        progProb = 1
        numOfDiscs = 2
        for disc in progression:
            if disc == 'B':
                progProb *= (1 / numOfDiscs)
            else:
                progProb *= ((numOfDiscs - 1) / numOfDiscs)
            numOfDiscs += 1
        winProb += progProb

print('Win Probability:', winProb)
print('Max win amount possible before loss:', int(1 / winProb))
```
Running the code gives us,
```
Win Probability: 0.00044063946502124476
Max win amount possible before loss: 2269
0.146565 seconds.
```
Thus, the maximum amount that can be allocated for a 15-turn game is **£2269**.