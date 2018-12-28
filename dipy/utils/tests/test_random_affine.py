""" Testing random affine module.
"""

from dipy.utils.random_affine import (generate_unit_determinant_matrix,
                                      generate_random_affine)


def main():
    mat = generate_unit_determinant_matrix()
    affine = generate_random_affine()


if __name__ == "__main__":
    main()
