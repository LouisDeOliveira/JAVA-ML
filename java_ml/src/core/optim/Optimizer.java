package core.optim;

import core.math.linalg.Matrix;

public abstract class Optimizer {
    public abstract void step(Matrix input, Matrix y_true, Matrix y_pred);
}
