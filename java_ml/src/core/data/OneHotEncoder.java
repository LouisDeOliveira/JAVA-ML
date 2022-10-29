package core.data;

import core.math.linalg.Matrix;

public class OneHotEncoder {
    private int NClasses;

    public OneHotEncoder(int NClasses) {
        this.NClasses = NClasses;
    }

    public Matrix encode(int classNumber) {
        Matrix encoded = new Matrix(NClasses, 1);
        encoded.setValue(classNumber, 0, 1.0);
        return encoded;
    }

    public int decode(Matrix output) {
        int classNumber = 0;
        double max = 0.0;
        for (int i = 0; i < NClasses; i++) {
            if (output.getValue(i, 0) > max) {
                max = output.getValue(i, 0);
                classNumber = i;
            }
        }
        return classNumber;
    }

}
