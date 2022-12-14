package core.math.linalg;

import java.io.Serializable;

import core.math.Function;

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

    public static Matrix ones(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = 1d;
            }
        }
        return m;
    }

    public static Matrix ones(int[] size) {
        return ones(size[0], size[1]);
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
        m.reshape(3, 2);
        System.out.println(m);
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

    public Matrix unRaveled() {
        // Unravels a matrix into a vector
        Matrix m = new Matrix(rows * cols, 1);
        int k = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[k++][0] = data[i][j];
            }
        }
        return m;
    }

    public void unRavel() {
        double[][] newData = new double[rows * cols][1];
        int k = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newData[k++][0] = data[i][j];
            }
        }
        this.rows = rows * cols;
        this.cols = 1;
        this.data = newData;

    }

    public Matrix reshaped(int rows, int cols) {
        if (rows * cols != this.rows * this.cols) {
            throw new IllegalArgumentException("Cannot reshape matrix to given dimensions.");
        }
        Matrix m = unRaveled();
        Matrix reshaped = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                reshaped.data[i][j] = m.data[i * cols + j][0];
            }
        }

        return reshaped;
    }

    /**
     * Reshapes in place
     * 
     * @param rows
     * @param cols
     */
    public void reshape(int rows, int cols) {
        if (rows * cols != this.rows * this.cols) {
            throw new IllegalArgumentException("Cannot reshape matrix to given dimensions.");
        }

        Matrix m = unRaveled();
        double[][] newData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newData[i][j] = m.data[i * cols + j][0];
            }
        }

        this.rows = rows;
        this.cols = cols;
        this.data = newData;
    }

    public Matrix copy() {
        Matrix m = like(this);

        for (int i = 0; i < rows; i++) {
            m.data[i] = data[i].clone();
        }

        return m;
    }

    public Matrix dot(Matrix m) {
        if (!compatible(m)) {
            throw new SizeMismatchException(this.getSize(), m.getSize(), "product");
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
    public boolean compatible(Matrix m) {
        return this.cols == m.getRows();
    }

    public boolean sameSize(Matrix m) {
        return this.rows == m.getRows() && this.cols == m.getCols();
    }

    public Matrix add(Matrix m) {
        if (!sameSize(m)) {
            throw new SizeMismatchException(this.getSize(), m.getSize(), "addition");
        } else {
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.cols; j++) {
                    result.data[i][j] = this.data[i][j] + m.data[i][j];
                }
            }
            return result;
        }
    }

    public Matrix substract(Matrix m) {
        if (!sameSize(m)) {
            throw new SizeMismatchException(this.getSize(), m.getSize(), "subtraction");
        } else {
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.cols; j++) {
                    result.data[i][j] = this.data[i][j] - m.data[i][j];
                }
            }
            return result;
        }
    }

    public Matrix elementWiseProduct(Matrix m) {
        if (!sameSize(m)) {
            throw new SizeMismatchException(this.getSize(), m.getSize(), "element-wise product");
        } else {
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                for (int j = 0; j < this.cols; j++) {
                    result.data[i][j] = this.data[i][j] * m.data[i][j];
                }
            }
            return result;
        }
    }

    public Matrix scale(double lambda) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] * lambda;
            }
        }
        return result;
    }

    public Matrix slice(int[] rows, int[] cols) {
        Matrix result = new Matrix(rows[1] - rows[0], cols[1] - cols[0]);
        for (int i = rows[0]; i < rows[1]; i++) {
            for (int j = cols[0]; j < cols[1]; j++) {
                result.data[i - rows[0]][j - cols[0]] = this.data[i][j];
            }
        }
        return result;
    }

    public double unitMatrixAsDouble() {
        if (rows == 1 && cols == 1) {
            return data[0][0];
        } else {
            throw new IllegalArgumentException("Matrix is not a unit matrix");
        }
    }

    public double sum() {
        double sum = 0d;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
        }
        return sum;
    }

    public double mean() {
        return sum() / (rows * cols);
    }

    public double max() {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > max) {
                    max = data[i][j];
                }
            }
        }
        return max;
    }

    public double min() {
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] < min) {
                    min = data[i][j];
                }
            }
        }
        return min;
    }

    public double norm(int dim) {
        double sum = 0d;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += Math.pow(data[i][j], dim);
            }
        }

        return Math.pow(sum, 1d / dim);
    }

    public Matrix map(Function<Double> f) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = f.f(this.data[i][j]);
            }
        }
        return result;
    }

    public Matrix clip(double min, double max) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[i][j] = Math.min(Math.max(this.data[i][j], min), max);
            }
        }
        return result;
    }
}