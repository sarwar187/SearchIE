from __future__ import division
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


# from plotting import line_chrt

class InformativePriorLogisticRegression(object):
    """Logistic regression with general spherical Gaussian prior.

    Arguments:
        w0 (ndarray, shape = (n_features,)): coefficient prior
        b0 (float): bias prior
        reg_param (float): regularization parameter $\lambda$
    """

    def __init__(self, w0, b0, reg_param):
        self.w0 = np.asarray(w0).flatten()  # Prior coefficients
        self.b0 = b0  # Prior bias
        self.reg_param = reg_param  # Regularization parameter (lambda)
        self.w = np.zeros(w0.shape)  # Learned w
        self.b = 0  # Learned b


    def fit(self, X, y, weights):
        """Fit the model according to the given training data.
        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        # self.w = np.append(self.w, self.b)
        # self.w0 = np.append(self.w0, self.b0)
        wb = np.append(self.w, self.b)
        # print "score"
        # print self.objective(wb, X, y)
        # print self.objective_grad(wb, X, y)
        # p = fmin_l_bfgs_b(self.objective, x0=self.w, args=(X, y), fprime=self.objective_grad)
        # p, q, r = fmin_l_bfgs_b(self.objective, x0=self.w, args=(X, y), approx_grad=True)
        # print p
        # print type(p)
        # print p.shape
        # FIT IS EASY GIVEN THIS NICE FUNCTION
        p, q, r = fmin_l_bfgs_b(self.objective, x0=wb, args=(X, y, weights), fprime=self.objective_grad)
        self.w = p[0:-1]
        self.b = p[-1]

        # print q
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        # print "w shapw"
        # print self.w.shape
        # DOING THE BIAS TRICK
        wb = np.append(self.w, self.b)
        X = np.hstack((X, np.ones(X.shape[0]).reshape((X.shape[0], 1))))
        # COMPUTING SCORE BY DOING DOT PRODUCT
        scores = np.dot(wb, X.T)
        # EXPONENTIATING SCORES AND CONVERTING TO PROBABILITY
        scores_exp = np.exp(scores)
        scores_exp += 1
        scores = np.exp(scores)
        scores /= scores_exp
        # PROBABILITY THRESHOLDING
        scores = [1 if score >= 0.5 else -1 for score in scores]
        return scores

    def predict_proba(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        # print "w shapw"
        # print self.w.shape
        # DOING THE BIAS TRICK
        wb = np.append(self.w, self.b)
        X = np.hstack((X, np.ones(X.shape[0]).reshape((X.shape[0], 1))))
        # COMPUTING SCORE BY DOING DOT PRODUCT
        scores = np.dot(wb, X.T)
        # EXPONENTIATING SCORES AND CONVERTING TO PROBABILITY
        scores_exp = np.exp(scores)
        scores_exp += 1
        scores = np.exp(scores)
        scores /= scores_exp
        # PROBABILITY THRESHOLDING
        #scores = [1 if score >= 0.5 else -1 for score in scores]
        return scores

    def objective(self, wb, X, y, weights):
        """Compute the objective function
        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """

        ones = np.ones(X.shape[0]).reshape((X.shape[0], 1))
        #print 'inside objective function'
        #print X.shape
        #print ones.shape
        X = np.concatenate((X, ones), axis=1)
        #print X.shape
        #X = np.hstack((X, np.ones(X.shape[0]).reshape((X.shape[0], 1))))
        print wb.shape
        print X.T.shape
        score = np.dot(wb, X.T)
        score = np.asarray(score).flatten()
        y = y.flatten()
        weights = weights.flatten()
        #print y.shape
        #print weights.shape
        #print score.shape
        score = np.log(1 + np.exp(-1 * y * score * weights))
        #print type(self.w0)
        #print wb.shape
        #print type(wb)
        wb0 = np.append(self.w0, self.b0)
        # IN THIS OBJECTIVE FUNCTION WE ARE COMBINING THE INFORMATIVE PRIOR. BEFORE THAT APPLIED BIAS TRICK
        sum_score = np.sum(score) + self.reg_param * (np.sum((wb - wb0) * (wb - wb0)))
        return sum_score

    def objective_grad(self, wb, X, y, weights):
        """Compute the derivative of the objective function
        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters: wb = [w,b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        X = np.hstack((X, np.ones(X.shape[0]).reshape((X.shape[0], 1))))
        num_train = X.shape[0]
        # print "shape "  + str(X.shape[1]) + "\t" + str(y.shape) + "\t" + num_train
        score = np.dot(wb, X.T)
        # print wb.shape
        dW = np.zeros(wb.shape)
        # RUN THE LOOP FOR ALL THE TRAINING SAMPLES. UPDATE THE GRADIENT VECTOR FOR EACH OF THEM
        #
        # print type(y)
        # print type(X)
        # print type(weights)
        # print type(score)
        #
        # print y.shape
        # print X.shape
        # print weights.shape
        # print score.shape
        score = np.asarray(score).flatten()
        for i in range(num_train):
            X_temp = -1 * y[i] * X[i] * weights[i]
            X_temp /= (1 + np.exp(1 * y[i] * score[i] * weights[i]))
            dW += np.asarray(X_temp).flatten()
        wb0 = np.append(self.w0, self.b0)
        dW += self.reg_param * 2 * (wb - wb0)
        # dW/=num_train
        return dW

    def get_params(self):
        """Get parameters for the model.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """

        return self.w, self.b

    def set_params(self, w, b):
        """Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
            reg_param (float): regularization parameter $\lambda$ (default: 0)
        """
        self.w = w
        self.b = b


def main():
    np.random.seed(0)

    # Example src for loading data
    train_X = np.load('../Data/q2_train_X.npy')
    train_y = np.load('../Data/q2_train_y.npy')
    test_X = np.load('../Data/q2_test_X.npy')
    test_y = np.load('../Data/q2_test_y.npy')
    w_prior = np.load('../Data/q2_w_prior.npy').squeeze()
    b_prior = np.load('../Data/q2_b_prior.npy')
    # print train_X.shape
    # print train_y.shape
    # cls = InformativePriorLogisticRegression(w_prior, b_prior, 1)
    # cls.fit(train_X, train_y)
    # return
    # CREATING TRAINING DATA CASES TO CHECK HOW BETTER THE INFORMATIVE PRIOR IS WITH LESS DATA
    training_data_cases = np.arange(10, 410, 10)
    accuracy_array = np.zeros((2, training_data_cases.shape[0]))
    # print accuracy_array.shape
    lambdas = [0, 10]
    # print training_data_cases
    # FOR DIFFERENT TRAINING SETS MEASURING THE ACCURACY AND PUTTING THEM IN ACCURACY_ARRAY
    for index, l in enumerate(lambdas):
        for index1, num_training in np.ndenumerate(training_data_cases):
            # print "num " + str(num_training)
            # print range(num_training)
            temp_train_X = train_X[range(num_training)]
            temp_train_y = train_y[range(num_training)]
            cls = InformativePriorLogisticRegression(w_prior, b_prior, l)
            cls.fit(temp_train_X, temp_train_y)
            scores = cls.predict(test_X)
            accuracy_array[index, index1] = np.sum(test_y == scores) / test_y.shape[0]
    # PRINTING AND PLOTTING. BUT THESE FUNCTIONS ARE NOT ALLOWED IN GRADESCOPE. SO COMMENTING THEM OUT
    # print accuracy_array
    # line_chrt(accuracy_array[0], accuracy_array[1], training_data_cases)
    # print test_y
    # print scores
    # print test_X.shape
    # print test_y.shape
    # print w_prior.shape
    # print b_prior.shape


if __name__ == '__main__':
    main()