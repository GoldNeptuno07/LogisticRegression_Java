package Models;

import Models.algebra_toolkit;

public class LogisticRegression
{
    // Initialize an attribute to define the input dimension
    int input_dim;
    // Define the learning rate
    double eta;
    // Initialize a variable to store the loss
    double loss = 0;
    // Initialize variables to store the gradients
    double[][] dW;
    double[][] dB;

    // Model parameters
    double[][] weights;
    double[][] bias = new double[1][1];

    // Create an objecto of the algebra_toolkit
    algebra_toolkit tools = new algebra_toolkit();

    public LogisticRegression(int input_dim, double learning_rate)
    {
        // Store the input dimention of the dataset
        this.input_dim = input_dim;
        // Define learning rate
        this.eta = learning_rate;

        // Initialize the weights from a normal distribution and the bias with zero
        int[] weights_shape = {this.input_dim, 1};
        this.weights = tools.randomNormal(weights_shape, 0, 1);
        this.bias[0][0] = 0;
    }

    public double[][] sigmoid(double[][] X)
    {
        // Compute the sigmoid function for each sample
        for(int i = 0; i < X.length; i++)
        {
            X[i][0] = 1 / (1 + Math.exp(-X[i][0]));
        }
        return X;
    }

    public double[][] predict(double[][] X)
    {
        // Define a 2D-array to store the result
        double[][] outputs = tools.sum(tools.matmul(X, this.weights), bias);
        return sigmoid(outputs);
    }

    private void compute_gradients(double[][] y_true, double[][] y_pred, double[][] X) throws Exception {
        // Compute the squared difference between the true labels and the predictions
        double[][] diff = tools.diff(y_pred, y_true);

        // Compute weights gradients
        double[][] dW = tools.mult(diff, X);
        dW = tools.reduce_mean(dW);
        this.dW = tools.transpose(dW);

        // Compute bias gradients
        this.dB = tools.reduce_mean(diff);

        // Store the current loss to display it
        this.loss = this.dB[0][0];
    }

    private void updateParameters() throws Exception {
        // Define the learning rate as 2D-array to performe the multiplication between the gradients and the learnin rate
        double[][] eta_array = {{this.eta}};

        // Update the weights
        double[][] aux = tools.mult(eta_array, this.dW);
        this.weights = tools.subtract(this.weights, aux);
        // Update the bias
        this.bias = tools.subtract(this.bias, tools.mult(eta_array, this.dB));
    }

    private void binary_crossentropy(double[][] y_true, double[][] y_pred)
    {
        // Reset the loss to zero
        this.loss = 0;
        // Value to avoid log of zero
        double eps = 1e-8;

        // Compute the binary-crossentropy for each sample
        for(int i = 0; i < y_true.length; i++)
        {
            this.loss += y_true[i][0] * Math.log10(y_pred[i][0] + eps) + (1 - y_true[i][0]) * Math.log10(1 - y_pred[i][0] + eps);
        }
        // Divide the loss by the number of samples
        this.loss /= -y_true.length;
    }

    private void train_step(double[][] X, double[][] y) throws Exception {

        // Performe a prediction
        double[][] prediction = predict(X);
        // Compute the loss to calculate the gradients
        compute_gradients(y, prediction, X);
        // Update the parameters
        updateParameters();
        // Compute the loss
        binary_crossentropy(y, prediction);
    }

    public void fit(double[][] X, double[][] y, int epochs) throws Exception {
        for(int i = 0; i < epochs; i++)
        {
            train_step(X, y);
            System.out.printf("Epoch. " + (i + 1) + "\tLoss. " + this.loss + "\n");
        }
    }
}