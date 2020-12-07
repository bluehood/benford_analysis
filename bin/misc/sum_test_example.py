import math

unnormalised = []
denom = 0
for n in range(10, 100):
    denom += n*math.log10(1+1/n)
    unnormalised.append(n*math.log10(1+1/n))
    
for n in range(10, 100):
    #print(f"{n}&{unnormalised[n - 10]}&{unnormalised[n - 10] / denom}\\\\")
    print("{}&{:.4f}&{:.4f}\\\\".format(n, unnormalised[n - 10], unnormalised[n - 10] / denom))