package core.nn;

import core.math.linalg.*;

public abstract class Loss {

    /**
     * Compute the loss between the predicted and target values.
     * 
     * @param y_pred The predicted values.
     * @param y_true The target values.
     * @return The loss.
     */
    public abstract double f(Matrix y, Matrix y_hat);

    /**
     * Compute the gradient of the loss with respect to the predicted values.
     * 
     * @param y_pred The predicted values.
     * @param y_true The target values.
     * @return The derivative of the loss.
     */
    public abstract Matrix df(Matrix y, Matrix y_hat);
}