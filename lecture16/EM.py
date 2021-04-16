import math

def N(x, mu, sigma2, d):
    term1 = 1 / ((2*math.pi*sigma2)**(d/2))
    pow1 = -1/(2*sigma2)
    pow2 = abs(x - mu)**2
    term2 = math.exp(pow1 * pow2)
    print('N:', term1 * term2)
    return term1 * term2

def pdf(x, mu, sigma2):
    pownum = -1 * (x - mu)**2
    powden = 2 * sigma2
    num = math.exp(pownum/powden)
    den = math.sqrt(sigma2) * math.sqrt(2*math.pi)
    result = num / den
    print('pdf:', result)
    return result

def pji(pj, x, mu, sigma2, d):
    thisN = N(x, mu, sigma2, d)
    thisPDF = pdf(x, mu, sigma2)
    return (pj*thisN)/thisPDF

pj = 0.5
x = 0.2
mu = -3
sigma2 = 4
d = 1

print(pji(pj, x, mu, sigma2, d))
