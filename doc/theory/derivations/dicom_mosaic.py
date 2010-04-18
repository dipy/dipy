''' Just showing the mosaic simplification '''
import sympy
from sympy import Matrix, Symbol, symbols, zeros, ones, eye

def numbered_matrix(nrows, ncols, symbol_prefix):
    return Matrix(nrows, ncols, lambda i, j: Symbol(
            symbol_prefix + '_{%d%d}' % (i+1, j+1)))

def numbered_vector(nrows, symbol_prefix):
    return Matrix(nrows, 1, lambda i, j: Symbol(
            symbol_prefix + '_{%d}' % (i+1)))


RS = numbered_matrix(3, 3, 'rs')

mdc, mdr, rdc, rdr = symbols(
    'md_{cols}', 'md_{rows}', 'rd_{cols}', 'rd_{rows}')

md_adj = Matrix((mdc - 1, mdr - 1, 0)) / -2
rd_adj = Matrix((rdc - 1 , rdr - 1, 0)) / -2

adj = -(RS * md_adj) + RS * rd_adj
adj.simplify()

Q = RS[:,:2] * Matrix((
        (mdc - rdc) / 2,
        (mdr - rdr) / 2))
Q.simplify()

assert adj == Q
