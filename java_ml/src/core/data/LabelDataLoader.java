package core.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.ArrayList;

import core.math.linalg.Matrix;

public class LabelDataLoader extends DataLoader {
    private ArrayList<Matrix> data;
    private OneHotEncoder encoder;

    public LabelDataLoader(int NClasses) {
        encoder = new OneHotEncoder(NClasses);
    }

    @Override
    public void readData(String dataPath) {
        try {
            DataInputStream dataStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataPath)));
            int magicNumber = dataStream.readInt();
            int numberOfItems = dataStream.readInt();
            data = new ArrayList<Matrix>();
            System.out.println("Reading data...");
            System.out.println("Magic number: " + magicNumber);
            System.out.println("Number of items: " + numberOfItems);

            for (int i = 0; i < numberOfItems; i++) {
                int classNumber = dataStream.readUnsignedByte();
                Matrix label = encoder.encode(classNumber);

                if (i % 10000 == 0 || i == numberOfItems - 1) {
                    System.out.println("Read " + ++i + " labels");
                }
                data.add(label);
            }
            dataStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public Matrix getData(int index) {
        return data.get(index).unRaveled();
    }

    @Override
    public int length() {
        return data.size();
    }

    public static void main(String[] args) {
        LabelDataLoader loader = new LabelDataLoader(10);
        loader.readData("./java_ml/data/train-labels.idx1-ubyte");
        System.out.println(loader.getData(0));
    }
}
