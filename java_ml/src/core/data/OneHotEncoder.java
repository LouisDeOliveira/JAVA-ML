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

}
