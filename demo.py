computing_power = [0.1, 0.2, 0.3, 0.4]
real_computing_power = [1 / item_power for item_power in computing_power]
sum_power = sum(real_computing_power)
power = [item_power / sum_power for item_power in real_computing_power]
length = [int(100 * item_power) for item_power in power]

print(length)
