def test_func(i):
    print(i)
    yield i+1

i = 0
print(i)
for idx in range(3):
    test_func(i)
