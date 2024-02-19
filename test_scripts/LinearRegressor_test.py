
from LinearRegressor import LinearRegressor




def test_initialise():
    lr=LinearRegressor(learning_rate=0.1,
                    error_threshold=2,
                    tolerance=5,
                    weights=[1,3,4])

    assert lr.is_fitted == False



