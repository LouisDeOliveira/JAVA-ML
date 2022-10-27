package core.math;

import core.math.linalg.Matrix;

public abstract class Function {
    public abstract double f(double x);

    public abstract double df(double x);

    /**
     * Applies the function to each element of the matrix.
     * 
     * @param x the matrix to apply the function to
     * @return the resulting matrix
     */
    public Matrix f(Matrix x) {
        Matrix y = new Matrix(x.getRows(), x.getCols());
        for (int i = 0; i < x.getRows(); i++) {
            for (int j = 0; j < x.getCols(); j++) {
                y.setValue(i, j, f(x.getValue(i, j)));
            }
        }
        return y;
    }

    /**
     * Applies the derivative of the function to each element of the matrix.
     * 
     * @param x the matrix to apply the derivative of the function to
     * @return the resulting matrix
     */
    public Matrix df(Matrix x) {
        Matrix y = new Matrix(x.getRows(), x.getCols());
        for (int i = 0; i < x.getRows(); i++) {
            for (int j = 0; j < x.getCols(); j++) {
                y.setValue(i, j, df(x.getValue(i, j)));
            }
        }
        return y;
    }

}
