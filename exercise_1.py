
import tensorflow as tf

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


def simple_linear_regression(input_feature, output):
    n = len(input_feature)
    sumxi = sum(input_feature)
    sumyi = sum(output)
    sumxi2 = sum(x * x for x in input_feature)
    sumxiyi = sum(x * y for x, y in zip(input_feature, output))

    slope = (sumxiyi-sumxi*sumyi/n)/(sumxi2-sumxi*sumxi/n)
    intercept = sumyi/n - slope*sumxi/n
    return intercept, slope


def get_regression_predictions(input_feature, intercept, slope):
    return input_feature * slope + intercept


def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    residual = get_regression_predictions(input_feature, intercept, slope) - output
    return sum(residual * residual)


def inverse_regression_predictions(output, intercept, slope):
    return (output - intercept) / slope


def graphlab_exercise():
    sales = graphlab.SFrame('kc_house_data.gl/')

    train_data,test_data = sales.random_split(.8,seed=0)

    sqft_data = train_data['sqft_living']
    price_data = train_data['price']

    sqft_intercept, sqft_slope = simple_linear_regression(sqft_data, price_data)
    print('intercept, slope', sqft_intercept, sqft_slope)
    answer1 = get_regression_predictions(2650, sqft_intercept, sqft_slope)
    print('predicted price for 2650 sqft', answer1)

    rss_training = get_residual_sum_of_squares(sqft_data, price_data, sqft_intercept, sqft_slope)
    answer2 = rss_training
    print('rss for training data', '%g'%answer2)

    answer3 = inverse_regression_predictions(800000, sqft_intercept, sqft_slope)
    print('est. sqft for $800k', answer3)

    bedroom_data = train_data['bedrooms']
    bedroom_intercept, bedroom_slope = simple_linear_regression(bedroom_data, price_data)

    sqft_test = test_data['sqft_living']
    bedroom_test = test_data['bedrooms']
    price_test = test_data['price']

    sqft_test_rss = get_residual_sum_of_squares(sqft_test, price_test, sqft_intercept, sqft_slope)
    bedroom_test_rss = get_residual_sum_of_squares(bedroom_test, price_test, bedroom_intercept, bedroom_slope)

    print ('sqft test rss', sqft_test_rss)
    print ('bedroom test rss', bedroom_test_rss)
    print ('ratio', sqft_test_rss/bedroom_test_rss)

def read_csv(file_name):
    reader = tf.TextLineReader(skip_header_lines=1)
    filename_queue = tf.train.string_input_producer([file_name])
    key, value = reader.read(filename_queue)
    return key, value

print(read_csv('data/kc_house_data.csv'))