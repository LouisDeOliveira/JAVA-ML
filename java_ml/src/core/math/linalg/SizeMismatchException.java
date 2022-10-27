package core.math.linalg;

public class SizeMismatchException extends IllegalArgumentException {
    public SizeMismatchException() {
        super();
    }

    public SizeMismatchException(String message) {
        super(message);
    }

    public SizeMismatchException(int[] size1, int[] size2) {
        super("Size mismatch: " + size1[0] + " x " + size1[1] + " and " + size2[0] + " x " + size2[1]);
    }
}
