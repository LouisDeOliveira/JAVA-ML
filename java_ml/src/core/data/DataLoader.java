package core.data;

import core.math.linalg.Matrix;

public abstract class DataLoader {
    public abstract void readData(String dataPath);

    public abstract Matrix getData(int index);

    public abstract int length();
}
