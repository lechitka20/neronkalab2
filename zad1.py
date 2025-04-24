from random import randint

spisok1 = [];
for i in range(20):
    spisok1.append(randint(0, 1000))
print(spisok1)
summa = 0
for i in spisok1:
    if i%2==0:
        summa+=i
print(summa)