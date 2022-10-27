package test.core.math.linalg;

import org.junit.*;

import core.math.linalg.Matrix;

public class MatrixTest {
    private Matrix m;
    private Matrix v;

    @Before
    public void setUp() {
        m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        v = new Matrix(new double[][] { { 1, 2, 3 } });
    }

    @Test
    public void testEquals() {
        Matrix mCopy = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Assert.assertTrue(m.equals(mCopy));
        mCopy.setValue(0, 0, 1.2);
        Assert.assertFalse(m.equals(mCopy));
    }

    @Test
    public void testGetCol() {
        Matrix mCol = m.getCol(0);
        Assert.assertTrue(mCol.equals(new Matrix(new double[][] { { 1 }, { 4 }, { 7 } })));
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
        Assert.assertTrue(rows == 3);
    }

    @Test
    public void testGetSize() {
        int[] size = m.getSize();
        int[] expectedSize = { 3, 3 };
        Assert.assertArrayEquals(size, expectedSize);
    }

    @Test
    public void testGetValue() {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                Assert.assertTrue(m.getValue(i, j) == i * m.getCols() + j + 1);
            }
        }

    }

    @Test
    public void testIdentity() {
        Matrix mIdentity = Matrix.identity(3);
        Assert.assertTrue(mIdentity.equals(new Matrix(new double[][] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } })));
    }

    @Test
    public void testDotIdentity() {
        Matrix mIdentity = Matrix.identity(3);
        Matrix mDot = m.dot(mIdentity);
        Assert.assertTrue(mDot.equals(m));
    }

    @Test
    public void testDotVector() {
        Matrix mDot = m.dot(v.transposed());
        Assert.assertTrue(mDot.equals(new Matrix(new double[][] { { 14 }, { 32 }, { 50 } })));
    }

    @Test
    public void testDotProduct() {
        Matrix mDot = m.dot(m);
        Assert.assertTrue(
                mDot.equals(new Matrix(new double[][] { { 30, 36, 42 }, { 66, 81, 96 }, { 102, 126, 150 } })));
    }

    @Test
    public void testDotVectorVector() {
        Matrix mDot = v.dot(v.transposed());
        Assert.assertTrue(mDot.equals(new Matrix(new double[][] { { 14 } })));
    }

    @Test
    public void testLike() {
        Matrix mLike = Matrix.like(m);
        Assert.assertArrayEquals(m.getSize(), mLike.getSize());

    }

    @Test
    public void testSetData() {
        double[][] data = new double[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        Matrix mSetData = Matrix.like(m);
        mSetData.setData(data);
        Assert.assertTrue(m.equals(mSetData));

    }

    @Test
    public void testCopy() {
        Matrix mCopy = m.copy();
        Assert.assertTrue(mCopy.equals(m));
        mCopy.setValue(1, 1, 69);
        Assert.assertFalse(mCopy.equals(m));
    }

    public static void main(String[] args) {
        Matrix m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        Matrix mCopy = m.copy();
        mCopy.setValue(1, 1, 69);
        System.out.println(m);
        System.out.println(mCopy);
    }

    @Test
    public void testSetValue() {
        m.setValue(0, 0, 69);
        Assert.assertTrue(m.getValue(0, 0) == 69);

    }

    @Test
    public void testTranspose() {
        Matrix m0 = m.copy();
        m0.transpose();
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                Assert.assertTrue(m0.getValue(i, j) == m.getValue(j, i));
            }
        }
        m0.transpose();
        Assert.assertTrue(m0.equals(m));
    }

    @Test
    public void testTransposed() {
        Matrix mT = m.transposed();
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                Assert.assertTrue(mT.getValue(i, j) == m.getValue(j, i));
            }
        }
        mT = mT.transposed();
        Assert.assertTrue(mT.equals(m));
    }

    @Test
    public void testZeros() {
        Matrix mZeros = Matrix.zeros(3, 3);
        for (int i = 0; i < mZeros.getRows(); i++) {
            for (int j = 0; j < mZeros.getCols(); j++) {
                Assert.assertTrue(mZeros.getValue(i, j) == 0);
            }
        }
    }

    @Test
    public void testZeros2() {
        Matrix mZeros = Matrix.zeros(m.getSize());
        for (int i = 0; i < mZeros.getRows(); i++) {
            for (int j = 0; j < mZeros.getCols(); j++) {
                Assert.assertTrue(mZeros.getValue(i, j) == 0);
            }
        }
    }
}
