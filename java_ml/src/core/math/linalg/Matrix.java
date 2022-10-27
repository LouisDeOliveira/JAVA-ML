package core.math.linalg;

import java.io.Serializable;

/**
 * A class for representing a matrix of doubles. Can be used for
 * vector as well as matrix operations. Similar to Numpy's ndarray.
 */
public class Matrix implements Serializable {

    public static Matrix like(Matrix m) {
        return new Matrix(m.rows, m.cols);
    }

    public static Matrix zeros(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = 0d;
            }
        }
        return m;
    }

    public static Matrix zeros(int[] size) {
        return zeros(size[0], size[1]);
    }

    public static Matrix identity(int size) {

        Matrix m = new Matrix(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    m.data[i][j] = 1d;
                } else {
                    m.data[i][j] = 0d;
                }
            }
        }
        return m;
    }

    public static void main(String[] args) {
        Matrix m = new Matrix(new double[][] { { 1, 2, 3 }, { 4, 5, 6 } });
        System.out.println(m);
        Matrix vec = m.getRow(0).transposed();
        System.out.println(vec.dot(m));

    }

    private double[][] data;

    private int rows;

    private int cols;

    private int[] size;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.size = new int[] { rows, cols };
        data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
        this.size = new int[] { rows, cols };
    }

    public Matrix(int[] size) {
        this.rows = size[0];
        this.cols = size[1];
        this.size = size;
        data = new double[rows][cols];
    }

    public void setData(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
        this.size = new int[] { rows, cols };
    }

    public void setValue(int row, int col, double value) {
        data[row][col] = value;
    }

    public double getValue(int row, int col) {
        return data[row][col];
    }

    public Matrix getRow(int row) {
        return new Matrix(new double[][] { data[row] });
    }

    public Matrix getCol(int col) {
        double[][] colData = new double[rows][1];
        for (int i = 0; i < rows; i++) {
            colData[i][0] = data[i][col];
        }
        return new Matrix(colData);
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public int[] getSize() {
        return size;
    }

    public double[][] getData() {
        return data;
    }

    /**
     * Trasposes this matrix in place.
     *
     */
    public void transpose() {
        double[][] newData = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newData[j][i] = data[i][j];
            }
        }
        data = newData;
        int temp = rows;
        rows = cols;
        cols = temp;
        size = new int[] { rows, cols };
    }

    /**
     * Returns a new matrix that is the transpose of this matrix.
     *
     * @return Transpose of this matrix.
     *
     */
    public Matrix transposed() {
        Matrix m = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[j][i] = data[i][j];
            }
        }
        return m;
    }

    public Matrix copy() {
        Matrix m = new Matrix(data);
        return m;
    }

    public Matrix dot(Matrix m) {
        if (!compatible(m)) {
            throw new SizeMismatchException(this.getSize(), m.getSize());
        } else {
            Matrix result = new Matrix(this.rows, m.getCols());
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < m.getCols(); j++) {
                    double sum = 0d;
                    for (int k = 0; k < this.cols; k++) {
                        sum += this.data[i][k] * m.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
            return result;
        }
    }

    @Override
    public String toString() {
        String s = "";
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                s += data[i][j] + " ";
            }
            s += "\n";

        }
        return s;
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }
        if (!(o instanceof Matrix)) {
            return false;
        }
        Matrix m = (Matrix) o;
        if (m.getRows() != rows || m.getCols() != cols) {
            return false;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (m.getValue(i, j) != data[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Returns true if this matrix can be right-multiplied with the given matrix.
     *
     * 
     */
    private boolean compatible(Matrix m) {
        return this.cols == m.getRows();
    }

}