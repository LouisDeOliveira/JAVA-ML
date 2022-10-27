package test.core.math.linalg;

import org.junit.*;

import core.math.linalg.Matrix;

public class MatrixTest {
    private Matrix m;

    @Before
    public void setUp() {
        m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });

    }

    @Test
    public void testCopy() {
        Matrix mCopy = m.copy();
        Assert.assertTrue(mCopy.equals(m));

    }

    @Test
    public void testEquals() {
        Matrix mCopy = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        Assert.assertTrue(m.equals(mCopy));
        mCopy.setValue(0, 0, 1.2);
        Assert.assertFalse(m.equals(mCopy));
    }

    @Test
    public void testGetCol() {
        Matrix mCol = m.getCol(0);
        Assert.assertTrue(mCol.equals(new Matrix(new double[][] { { 1 }, { 4 } })));
    }

    @Test
    public void testGetCols() {
        int cols = m.getCols();
        Assert.assertTrue(cols == 3);
    }

    @Test
    public void testGetRow() {
        Matrix mRow = m.getRow(0);
        Assert.assertTrue(mRow.equals(new Matrix(new double[][] { { 1, 2, 3 } })));
    }

    @Test
    public void testGetRows() {
        int rows = m.getRows();
        Assert.assertTrue(rows == 2);
    }

    @Test
    public void testGetSize() {

    }

    @Test
    public void testGetValue() {

    }

    @Test
    public void testIdentity() {

    }

    @Test
    public void testLike() {

    }

    @Test
    public void testMain() {

    }

    @Test
    public void testSetData() {

    }

    @Test
    public void testSetValue() {

    }

    @Test
    public void testToString() {

    }

    @Test
    public void testTranspose() {

    }

    @Test
    public void testTransposed() {

    }

    @Test
    public void testZeros() {

    }

    @Test
    public void testZeros2() {

    }
}
