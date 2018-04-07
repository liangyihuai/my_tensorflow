
a = [5, 4, 6, 8]
b = [3, 2, 1, 0]

z = zip(a, b)

# for i in z:
#     print(i)

start_zip = zip(*z)
for k in start_zip:
    print(k)





