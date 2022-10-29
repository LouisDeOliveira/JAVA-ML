package test.core.data;

import org.junit.*;
import core.math.linalg.Matrix;
import core.data.OneHotEncoder;

public class OneHotEncoderTest {
    private OneHotEncoder encoder;
    private Matrix output;

    @Before
    public void setUp() {
        int NClasses = 3;
        encoder = new OneHotEncoder(NClasses);
        output = new Matrix(new double[][] { { 0.1 }, { 0.2 }, { 0.7 } });
    }

    @Test
    public void testEncode() {
        int classNumber = 2;
        Matrix encoded = encoder.encode(classNumber);
        Matrix expected = new Matrix(new double[][] { { 0.0 }, { 0.0 }, { 1.0 } });
        Assert.assertTrue(encoded.equals(expected));
    }

    @Test
    public void testDecode() {
        int classNumber = encoder.decode(output);
        Assert.assertTrue(classNumber == 2);
    }

}
