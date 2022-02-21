import numpy as np
from numpy import zeros
from dipy.segment.threshold import upper_bound_by_percent, upper_bound_by_rate
from numpy.testing import assert_equal


def test_adjustment():

    imga = zeros([128, 128])

    for y in range(128):
        for x in range(128):
            if y > 10 and y < 115 and x > 10 and x < 115:
                imga[x, y] = 100
            if y > 39 and y < 88 and x > 39 and x < 88:
                imga[x, y] = 150
            if y > 59 and y < 69 and x > 59 and x < 69:
                imga[x, y] = 255

    high_1 = upper_bound_by_rate(imga)
    high_2 = upper_bound_by_percent(imga)
    vol1 = np.interp(imga, xp=[imga.min(), high_1], fp=[0, 255])
    vol2 = np.interp(imga, xp=[imga.min(), high_2], fp=[0, 255])
    count2 = (88 - 40) * (88 - 40)
    count1 = (114 - 10) * (114 - 10)

    count1_test = 0
    count2_test = 0

    count2_upper = (88 - 40) * (88 - 40)
    count1_upper = (114 - 10) * (114 - 10)

    count1_upper_test = 0
    count2_upper_test = 0

    value1 = np.unique(vol1)
    value2 = np.unique(vol2)

    for i in range(128):
        for j in range(128):
            if vol1[i][j] > value1[1]:
                count2_test = count2_test + 1
            if vol1[i][j] > 0:
                count1_test = count1_test + 1

    for i in range(128):
        for j in range(128):
            if vol2[i][j] > value2[1]:
                count2_upper_test = count2_upper_test + 1
            if vol2[i][j] > 0:
                count1_upper_test = count1_upper_test + 1


    assert_equal(count2, count2_test)
    assert_equal(count1, count1_test)

    assert_equal(count2_upper, count2_upper_test)
    assert_equal(count1_upper, count1_upper_test)
