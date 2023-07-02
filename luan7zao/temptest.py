dpr = [0.1, 0.2, 0.3, 0.4, 0.5]
depths = (2, 2, 6, 2)

i_layer = 3
a = sum(depths[:i_layer])
b = sum(depths[:i_layer+1])
drop_path = dpr[a: b]

print(drop_path)