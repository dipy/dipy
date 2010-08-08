import rng

rng.ix, rng.iy, rng.iz, rng.it = 100001, 200002, 300003, 400004

N = 1000

a = [rng.WichmannHill2006() for i in range(N)]

'''
See the paper B.A. Wichmann, I.D. Hill, Generating good pseudo-random
numbers, Computational Statistics & Data Analysis, Volume 51, Issue 3,
1 December 2006, Pages 1614-1622, ISSN 0167-9473,DOI:10.1016/j.csda.2006.05.019.
DOI:10.1016/j.csda.2006.05.019.
(http://www.sciencedirect.com/science/article/B6V8V-4K7F86W-2/2/a3a33291b8264e4c882a8f21b6e43351)
for advice on generating many sequences for use together, and on
alternative algorithms and codes
'''
