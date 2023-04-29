---
layout: post
title: "#59 - XOR decryption"
date: 2017-06-20 09:46
number: 59
tags: [05_diff]
---
> Each character on a computer is assigned a unique code and the  preferred standard is ASCII (American Standard Code for Information Interchange). For example, uppercase A = 65, asterisk (\*) = 42, and  lowercase k = 107.
> 
> A modern encryption method is to take a text file, convert the bytes to ASCII, then XOR each byte with a given value, taken from a secret key. The advantage with the XOR function is that using the same encryption key on the cipher text, restores the plain text; for example, 65 XOR 42 = 107, then 107 XOR 42 = 65.
> 
> For unbreakable encryption, the key is the same length as the plain text message, and the key is made up of random bytes. The user would keep the encrypted message and the encryption key in different locations, and without both "halves", it is impossible to decrypt the message.
> 
> Unfortunately, this method is impractical for most users, so the modified method is to use a password as a key. If the password is shorter than the message, which is likely, the key is repeated cyclically throughout the message. The balance for this method is using a sufficiently long password key for security, but short enough to be memorable.
> 
> Your task has been made easy, as the encryption key consists of three lower case characters. Using [p059_cipher.txt](https://projecteuler.net/project/resources/p059_cipher.txt)  (right click and 'Save Link/Target As...'), a file containing the encrypted ASCII codes, and the knowledge that the plain text must contain common English words, decrypt the message and find the sum of the ASCII values in the original text.
{:.lead}
* * *

It seems straightforward: Try every combination of three lowercase letters, decrypt the message, and see if it makes sense. However, there are a couple of problems here:
* All combinations of three lowercase letters is $26^3=17576$, which might not be too bad in this problem, but can get unruly if we have any more letters.
* What does "if it makes sense" mean? It means "the excerpt is read properly and is legible", but how do we translate that logic into code?

For the first bullet point, we can take this approach: Since the sequence of three letters will be repeated, that means each letter in the key will decrypt every **third** letter in the encrypted message. We can examine if every third letter can indeed be translated into a letter, digit, or special character. If it's outside of this ASCII range, then we can rule out that letter in that position, and drastically reduce the number of possibilities we have to sift through. 

For the second point, note that the problem says that the text contains common English words. More quantitatively, we can conclude that **the distribution of the letters in the decrypted text follow closely with that of the English language**. Now, the distribution of the letters in the English language can be seen at [this Wikipedia page](https://en.wikipedia.org/wiki/Letter_frequency). To measure "closeness", we can calculate the character-wise distance this frequency vector to that of our decrypted message. The key which produces the smallest distance to the ground truth distribution will be the correct key.

In Python, we have `chr()` and `ord()` which goes back and forth between the character and the integer it represents. The XOR operation in Python is the single carat `^`. The valid ASCII range in our problem is from 32 (space) to 122. We use `itertools.product` to go through all combinations of letters. 
```python
# file: "problem059.py"
with open('p059_cipher.txt') as f:
    encryptedMess = [int(a) for a in f.readline().split(',')]

# Find the only possible letters that could
# be part of the enccryption key.
encryptedPosses = []
for i in range(3):
    possLetters = []
    for letter in string.ascii_lowercase:
        intRep = ord(letter)
        # Encrypt every 3rd letter starting
        # from i...
        decryptedCharas = [intRep ^ encInt for encInt in encryptedMess[i::3]]
        # All of the decrypted charas have to be
        # letters, digits, special charas, or spaces.
        # So integer is between 32 and 122.
        if all(32 <= charaInt <= 122 for charaInt in decryptedCharas):
            possLetters.append(letter)
    encryptedPosses.append(possLetters)

# Ground truth frequency for each letter
# in english language (from Wikipedia)
groundTruth = [0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
               0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
               0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
               0.00978, 0.02360, 0.00150, 0.01974, 0.00074]
bestKey = ''
closestDist = float('inf')
for possibleKey in product(*encryptedPosses):
    # Decrypt text given this key
    decryptedText = [chr(ord(possibleKey[i % 3]) ^ encryptedMess[i]) for i in range(len(encryptedMess))]
    # Turn everything to lowercase and grab
    # only letters...
    onlyLetters = [chara.lower() for chara in decryptedText if chara.lower() in string.ascii_lowercase]
    # Calculate counts of each letter
    letterCounts = np.zeros(26)
    for letter in onlyLetters:
        letterCounts[ord(letter) - 97] += 1
    # Calculate frequency
    letterFreq = letterCounts / np.sum(letterCounts)
    # Calculate distance to ground truth
    # distribution
    distance = np.linalg.norm(letterFreq - groundTruth)
    if distance < closestDist:
        closestDist = distance
        bestKey = ''.join(possibleKey)

print('Best key is "{}".'.format(bestKey))

# Now decrypt message and add all
# integers...
decryptedSum = sum([ord(bestKey[i % 3]) ^ encryptedMess[i] for i in range(len(encryptedMess))])
print('Sum of decrypted integers: {}'.format(decryptedSum))
```
Running the entire code results in,
```
Best key is "god".
Sum of decrypted integers: 107359
0.07973290677218499 seconds.
```
Therefore, the key to the message is "god" and the sum is **107359**. One more robust change we can make is that we check the decrypted words against a separate dictionary. We were lucky that the smallest distribution distance resulted in the correct answer.
## Update July 23, 2019
Due to the religious nature of the key and the decrypted text, the encrypted message has been changed as of February 5, 2019. Instead, the code produces an output of
```
Best key is "exp".
Sum of decrypted integers: 129448
0.02513697772366312 seconds.
```
