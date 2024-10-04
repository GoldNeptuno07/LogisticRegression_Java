package Models;

public class LogisticRegression
{
    // Initialize an attribute to define the input dimension
    int input_dim;
    // Initialize an attribute to store the learning rate
    double eta;
    // Initialize an attribute to store the loss
    double loss = 0;
    // Initialize an attribute to store the gradients
    double[][] dW;
    double[][] dB;

    // Define model's parameters (weights & bias)
    double[][] weights;
    double[][] bias = new double[1][1];

    // Create an objecto of the algebra_toolkit class to perform algebraic operations
    algebra_toolkit tools = new algebra_toolkit();

    public LogisticRegression(int input_dim, double learning_rate)
    {
        // Store the input dimention of the dataset
        this.input_dim = input_dim;
        // Define learning rate
        this.eta = learning_rate;

        // Initialize the model's weights from a normal distribution and the bias with zero
        int[] weights_shape = {this.input_dim, 1};
        this.weights = tools.randomNormal(weights_shape, 0, 1);
        this.bias[0][0] = 0;
    }

    public double[][] sigmoid(double[][] X)
    {
        // Compute the sigmoid function for each sample in X
        for(int i = 0; i < X.length; i++)
        {
            X[i][0] = 1 / (1 + Math.exp(-X[i][0]));
        }
        return X;
    }

    public double[][] predict(double[][] X)
    {
        // Define a 2D-array to store the model's output computed
        double[][] Z = tools.sum(tools.matmul(X, this.weights), bias);
        return sigmoid(Z);
    }

    private void compute_gradients(double[][] y_true, double[][] y_pred, double[][] X) throws Exception {
        // Compute the difference between the true labels and the predicted labels
        double[][] diff = tools.subtract(y_pred, y_true);

        // Compute the gradients of the weights based on the loss
        double[][] dW = tools.mult(diff, X);
        dW = tools.reduce_mean(dW);
        this.dW = tools.transpose(dW);

        // Compute gradients of the bias based on the loss
        this.dB = tools.reduce_mean(diff);
    }

    private void updateParameters() throws Exception {
        // Define the learning rate as 2D-array to perform the multiplication between the gradients and the learnin rate
        double[][] eta_array = {{this.eta}};

        // Update the gradients of the weights
        double[][] aux = tools.mult(eta_array, this.dW);
        this.weights = tools.subtract(this.weights, aux);
        // Update the gradients of the bias
        this.bias = tools.subtract(this.bias, tools.mult(eta_array, this.dB));
    }

    private void binary_crossentropy(double[][] y_true, double[][] y_pred)
    {
        // Reset the loss to zero
        this.loss = 0;
        // Small value to avoid logarithm of zero during loss calculation
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

        // Store the predicted values for input X (forward propagation)
        double[][] prediction = predict(X);
        // Compute the gradients with respect to the weights and bias
        compute_gradients(y, prediction, X);
        // Update the model's parameters using the calculated gradients
        updateParameters();
        // Compute the loss between the true labels and the predicted labels
        binary_crossentropy(y, prediction);
    }

    public void fit(double[][] X, double[][] y, int epochs) throws Exception {
        for(int i = 0; i < epochs; i++)
        {
            // Perform a train step to tweak the model's parameters
            train_step(X, y);
            // Display the current epoch number and the computed loss
            System.out.printf("Epoch. " + (i + 1) + "\tLoss. " + this.loss + "\n");
        }
    }
}