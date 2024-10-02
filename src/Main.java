import Models.algebra_toolkit;
import Models.LogisticRegression;

public class Main
{
    public static void main(String[] args) throws Exception {
        // Create an object of the algebra_toolkit class
        algebra_toolkit tools = new algebra_toolkit();

        // Dataset dimension
        int[] shape = {10, 1};

        // Initialize random data from a normal distribution
        double[][] samples_0 = tools.randomNormal(shape, 0, 1);
        double[][] samples_1 = tools.randomNormal(shape, 5, 1);

        // Concatenate the data and fill the labels with 0s and 1s
        double[][] X_data = new double[shape[0]*2][shape[1]];
        double[][] y_data = new double[shape[0]*2][1];

        for(int i = 0; i < shape[0]; i++)
        {
            X_data[i] = samples_0[i];
            X_data[i + shape[0]] = samples_1[i];

            y_data[i][0] = 0;
            y_data[i + shape[0]][0] = 1;
        }

        // Initialize the classifier
        LogisticRegression classf = new LogisticRegression(shape[1], 0.01);
        classf.fit(X_data, y_data, 100);

        // Make a prediction
        double[][] prediction = classf.predict(X_data);

        // Print the resultant matrix C after dot-product operation
        int ROWS = prediction.length, COLUMNS= prediction[0].length;
        int output;
        for(int i = 0; i < ROWS; i++)
        {
            for(int j = 0; j < COLUMNS; j++)
            {
                output = Double.compare(prediction[i][j], 0.5);
                System.out.print( output + " ");
            }
            if (i == shape[0] - 1)
                System.out.println();
        }
    }
}