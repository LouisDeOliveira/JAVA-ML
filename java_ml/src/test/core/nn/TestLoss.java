package test.core.nn;

import org.junit.Assert;
import org.junit.Test;

import core.math.linalg.Matrix;
import core.nn.Loss;

public class TestLoss {

    @Test
    public void testMSE() {
        Matrix y_true = new Matrix(new double[][] { { 3, -0.5, 2, 7 } });
        Matrix y_pred = new Matrix(new double[][] { { 2.5, 0.0, 2, 8 } });
        y_true = y_true.transposed();
        y_pred = y_pred.transposed();

        double loss = Loss.MSE.f(y_true, y_pred);
        Assert.assertEquals(0.375, loss, 0.0001);
    }

    @Test
    public void testGradMSE() {
        Matrix y_true = new Matrix(new double[][] { { 1, 0, 0 } });
        Matrix y_pred = new Matrix(new double[][] { { 0, 1, 0 } });
        y_true = y_true.transposed();
        y_pred = y_pred.transposed();

        Matrix grad = Loss.MSE.df(y_true, y_pred);
        Matrix expected = y_pred.substract(y_true).scale((double) 2 / 3);

        Assert.assertEquals(expected, grad);

    }

}
