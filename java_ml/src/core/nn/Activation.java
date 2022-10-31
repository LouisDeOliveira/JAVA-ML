package core.nn;

import core.math.linalg.Matrix;

public interface Activation {
    public Matrix f(Matrix input);

    public Matrix df(Matrix input);
}
