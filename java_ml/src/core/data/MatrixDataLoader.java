package core.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.util.ArrayList;

import core.math.linalg.Matrix;

public class MatrixDataLoader extends DataLoader {
    private ArrayList<Matrix> data;

    @Override
    public void readData(String dataPath) {
        try {
            DataInputStream dataStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataPath)));
            int magicNumber = dataStream.readInt();
            int numberOfItems = dataStream.readInt();
            int numberOfRows = dataStream.readInt();
            int numberOfColumns = dataStream.readInt();
            data = new ArrayList<Matrix>();
            System.out.println("Reading data...");
            System.out.println("Magic number: " + magicNumber);
            System.out.println("Number of items: " + numberOfItems);
            System.out.println("Number of rows: " + numberOfRows);
            System.out.println("Number of columns: " + numberOfColumns);

            for (int i = 0; i < numberOfItems; i++) {
                Matrix item = new Matrix(numberOfRows, numberOfColumns);
                for (int j = 0; j < numberOfRows; j++) {
                    for (int k = 0; k < numberOfColumns; k++) {
                        double value = (double) dataStream.readUnsignedByte() / 255.0;
                        item.setValue(j, k, value);
                    }
                }
                if (i % 10000 == 0 || i == numberOfItems - 1) {
                    System.out.println("Read " + ++i + " items");
                }
                data.add(item);
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
        MatrixDataLoader loader = new MatrixDataLoader();
        loader.readData("./java_ml/data/train-images.idx3-ubyte");
        System.out.println(loader.getData(0));
    }

}
