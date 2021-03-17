import math

print(f'Verify that Benford\'s law is properly normalised by performing the sum over all probabilities.\n')

unnormalised = []
denom = 0
for n in range(10, 100):
    denom += math.log10(1+1/n)
    unnormalised.append(math.log10(1+1/n))
    
print(f'Sum Second Digit Test: {denom}')
# for n in range(10, 100):
#     #print(f"{n}&{unnormalised[n - 10]}&{unnormalised[n - 10] / denom}\\\\")
#     print("{}&{:.4f}&{:.4f}\\\\".format(n, unnormalised[n - 10], unnormalised[n - 10] / denom))