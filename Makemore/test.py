words = open('Makemore/names.txt', 'r').read().splitlines();

print(words[2:12]);

print(min(len(w)for w in words));

print(max(len(w)for w in words));

for w in words [:1]:
    for ch1,ch2 in zip(w,w[1:]):
        print(ch1,ch2)


