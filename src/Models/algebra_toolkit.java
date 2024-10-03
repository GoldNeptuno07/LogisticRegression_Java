package Models;

import java.util.Random;

public class algebra_toolkit
{
    public double[][] sum(double[][] A, double[][] B)
    {
        // Compute the number of rows and columns for the sum operation
        int ROWS = A.length, COLUMNS = A[0].length;

        // Perform the element-wise sum of matrix A and the bias vector B
        // Add each element of the bias B to the corresponding row in A
        for(int i = 0; i < ROWS; i++)
        {
            for(int j = 0; j < COLUMNS; j++)
            {
                A[i][j] += B[0][j];
            }
        }
        return A;
    }

    public double[][] subtract(double[][] A, double[][] B)
    {
        // Compute the numner of ROWS and COLUMNS of the matrix A
        int ROWS = A.length, COLUMNS = A[0].length;

        // Initialize a 2D-array to store the result of the operation
        double[][] C = new double[ROWS][COLUMNS];

        // Perform the subtraction operation between the matrix A and B
        // Subtract the model's parameters (weights | bias) with the computed gradients (dW | dB)
        for(int  i = 0; i < ROWS; i++)
        {
            for(int j = 0; j < COLUMNS; j++)
            {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    public double[][] mult(double[][] A, double[][] B) throws Exception {
        // Perform either :
        //              * the multiplication between a vector and a vector
        //              * the multiplication between a coeficient and vector
        int i, j, ROWS, COLUMNS;

        if (A.length == B.length) // ( vector | vector ) multiplication
        {
            // Initiliaze two 2D-arrays: X will store the higher-dimension vector and
            // B the lower dimension vector
            double[][] X;
            double[][] Y;

            // Let's verify if the 2D-arrays have the same dimension
            if (A[0].length >= B[0].length)
            {
                X = A;
                Y = B;
            }
            else
            {
                X = B;
                Y = A;
            }

            // Compute the dimensions of the 2D arrays
            ROWS = X.length;
            COLUMNS = X[0].length;

            // Define a 2D array to store operation's result
            double[][] C = new double[ROWS][COLUMNS];

            // Perform the ( array | array ) multiplication
            int ind;
            for (i = 0; i < ROWS; i++)
            {
                for(j = 0; j < COLUMNS; j++)
                {
                    ind = Math.min(j, 0);
                    C[i][j] = X[i][j] * Y[i][ind];
                }
            }

            return C;
        }
        else if (A.length == 1 || B.length == 1) // ( coeficient | array ) multiplication
        {
            // Get the coeficient to perform the operation
            double[][] coef = (A.length == 1) ? A : B;
            double[][] array = (A.length == 1) ? B : A;
            int idx;

            // Get the number of rows and columns of the 2D array
            ROWS = array.length;
            COLUMNS = array[0].length;

            // Define a 2D array to store operation's result
            double[][] C = new double[ROWS][COLUMNS];

            // Perform (coeficient | array ) multiplication
            for(i = 0; i < ROWS; i++)
            {
                for(j = 0; j < COLUMNS; j++)
                {
                    idx = Math.min(j, coef[0].length - 1);
                    C[i][j] = coef[0][idx] * array[i][j];
                }
            }

            return C;
        }
        else
        {
            throw new Exception("Invalid arrays size. A size " + A.length + " and B size " + B.length + ".");
        }
    }

    public double[][] matmul(double[][] A, double[][] B)
    {
        /*
         *   Method to perform the dot-product operation between two 2D-arrays.
         */

        // Compute the matrix size (square matrices)
        int X = A[0].length, Y = A.length, Z = B[0].length;

        // Define a 2D array to store the result of the operation
        double[][] C = new double[Y][Z];

        // Perform dot-product between A and B matrices (square matrices)
        for(int i = 0; i < X; i++)
        {
            for(int j = 0; j < Y; j++)
            {
                for(int z = 0; z < Z; z++)
                {
                    C[j][z] += A[j][i] * B[i][z];
                }
            }
        }

        return C;
    }

    public double[][] randomNormal(int[] shape, double mean, double std)
    {
        // Define a 2D array where we are going to store the samples
        double[][] samples = new double[shape[0]][shape[1]];

        // Create an object from the Random class
        Random rand = new Random();

        // Generate random samples from a normal distribution
        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                samples[i][j] = rand.nextGaussian(mean, std);
            }
        }

        return samples;
    }

    public double[][] reduce_sum(double[][] X)
    {
        // Define an integer to store the input dimension of the data
        int n = X[0].length;
        // Define a 2D-array to store the result
        double[][] result = new double[1][n];

        // Compute the sum along the axis = 0
        for (double[] x : X)
            for (int j = 0; j < n; j++) {
                result[0][j] += x[j];
            }

        return result;
    }

    public double[][] reduce_mean(double[][] X)
    {
        // Define an integer to store the input dimension
        int input_dim = X[0].length;
        int N = X.length;
        // Perform the sum along the rows
        double[][] addition = reduce_sum(X);
        // Divide the data by the amount of samples (N)
        for(int i = 0; i < input_dim; i++)
        {
            addition[0][i] /= N;
        }

        return addition;
    }

    public double[][] transpose(double[][] X)
    {
        // Initialize a 2D-array to storeh the transposed array
        double[][] C = new double[X[0].length][X.length];
        // Tranpose X and store the transposed result in C
        for(int i = 0; i < X.length; i++)
        {
            for(int j = 0; j < X[0].length; j++)
            {
                C[j][i] = X[i][j];
            }
        }

        return C;
    }
}