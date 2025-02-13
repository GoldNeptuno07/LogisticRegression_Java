package com.gui;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.Color;

import java.io.File;
import java.util.List;
import java.util.Map;

import com.tools.PipeLine;
import com.models.LogisticRegression;

/**
 * This class will be used to generate the main GUI, in order to define the layout of the widgets and
 * show the decision boundary of the Logistic Regression model.
 *
 * @author Angel Cruz
 *
 * */
public class GraphicInterface extends JFrame {
    // GUI widgets
    final private JFrame mainFrame;
    final private JMenuBar  menuBar;
    final private JLabel instructionLabel;
    private XYSeriesCollection dataset; // Chart points

    // Decision boundary / model
    private LogisticRegression classif;
    private ChartPanel decisionBoundaryPanel;
    private double model_loss;
    private XYSeries prediction_series;

    // Data
    private String datasetURL;
    final private int numberFeatures = 2;
    // Preprocessing Pipeline
    PipeLine pipeline = new PipeLine();


    /**
     * Main constructor to define the main frame and all its widgets
     *
     * */
    public GraphicInterface()
    {
        /* Define Main Frame */
        mainFrame = new JFrame("Logistic Regression");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.setMinimumSize(new Dimension(600, 400));

        /* Define MenuBar Buttons */
        menuBar = new JMenuBar();
        JMenu fitModel = new JMenu("Fit Model");
        JMenu helpWindow = new JMenu("Help");

        JMenuItem selectDataBttn = new JMenuItem("Select Dataset");

        fitModel.add(selectDataBttn);
        menuBar.add(fitModel);
        menuBar.add(helpWindow);

        /* MenuBar Action Listener (Commands) */
        selectDataBttn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser chooserWindow = new JFileChooser();
                int returnValue = chooserWindow.showSaveDialog(null);

                // Select the data path
                if (returnValue == JFileChooser.APPROVE_OPTION) {
                    File selectedFile = chooserWindow.getSelectedFile();
                    if (selectedFile != null) {
                        datasetURL = selectedFile.getAbsolutePath();
                        try {
                            displayModelWidgets();
                        } catch (Exception ex) {
                            throw new RuntimeException(ex);
                        }
                    } else {
                        System.err.println("Error: No file selected.");
                    }
                } else {
                    System.err.println("User canceled file selection.");
                }
            }
        });

        /* Define Instruction Label */
        instructionLabel = new JLabel("\tPlease select a dataset to train the Logistic Regression model...");

        /* Adding the widgets to the Main Frame */
        mainFrame.add(menuBar, BorderLayout.NORTH);
        mainFrame.add(instructionLabel, BorderLayout.CENTER);

        mainFrame.setVisible(true);
    }

    /**
     * Method to train the model in order to plot the decision boundary and display
     * the text fields to make a prediction.
     * */
    private void displayModelWidgets() throws Exception {
        /* Build Decision Boundary Chart */
        boolean code= displayDecisionBoundary(true);
        if(!code)
        {
            return;
        }
        mainFrame.remove(instructionLabel);

        /* SOUTH Prediction Panel */
        JPanel predictionPanel = new JPanel();
        predictionPanel.setLayout(new FlowLayout());

        JLabel predLabel = new JLabel("Plot point. ");
        JTextField[] featuresFields = new JTextField[numberFeatures];

        predictionPanel.add(predLabel);
        for(int i = 0; i < numberFeatures; i++)
        {
            featuresFields[i] = new JTextField(10);
            predictionPanel.add(featuresFields[i]);
        }
        JButton predictBtnn = new JButton("Show"); // Predict Bttn
        predictBtnn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Double X, Y;
                if(featuresFields[0].getText().isEmpty() || featuresFields[1].getText().isEmpty()) return;

                X= Double.parseDouble(featuresFields[0].getText());
                Y= Double.parseDouble(featuresFields[1].getText());

                if(dataset.indexOf("New Samples") == -1)
                {
                    prediction_series= new XYSeries("New Samples");
                    prediction_series.add(X,Y);
                    dataset.addSeries(prediction_series);
                }
                else
                    prediction_series.add(X,Y);

                featuresFields[0].setText("");
                featuresFields[1].setText("");

                mainFrame.repaint();
                mainFrame.revalidate();
            }
        });

        // Retrain Model Button
        JButton retrain_model_bttn = new JButton("Retrain Model");
        retrain_model_bttn.setBackground(Color.BLACK);
        retrain_model_bttn.setForeground(Color.white);
        retrain_model_bttn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    // Build the decision boundary
                    displayDecisionBoundary(false);
                    // Add the decisionBoundary to the mainFrame
                    mainFrame.add(decisionBoundaryPanel);
                    mainFrame.revalidate();
                    mainFrame.repaint();

                } catch (Exception ex) {
                    throw new RuntimeException(ex);
                }
            }
        });


        /* Add widgets to the main frame */
        mainFrame.add(decisionBoundaryPanel);
        mainFrame.add(predictionPanel, BorderLayout.SOUTH);

        predictionPanel.add(predictBtnn);
        predictionPanel.add(retrain_model_bttn);

        mainFrame.pack();
    }

    /**
     * Method to build the decision boundary chart.
     * */
    private boolean displayDecisionBoundary(boolean executePipeLine) throws Exception {
        /*
        *   Execute the PipeLine
        * */
        if(executePipeLine)
        {
            boolean code= PipeLine.extractAndTransform(datasetURL);

            if(!code)
                return false;
        }
        Map<String, List<List<Double>>> data_map = pipeline.loadDataset();
        String[] header = pipeline.get_header();

        // Get and cast X and y set.
        double[][] X_samples = convertList2Array(data_map.get("X"));
        double[][] y_samples = convertList2Array(data_map.get("y"));

        /*
        * Train Model
        * */
        classif = new LogisticRegression(2, 0.01, 0);
        classif.fit(X_samples, y_samples, 20);
        model_loss= classif.get_loss();
        mainFrame.setTitle(String.format("Logistic Regression. (loss. %.2f)", model_loss));

        /*
        * Predicting Grid
        * */
        double[] min = {Double.MAX_VALUE,Double.MAX_VALUE};
        double[] max = {Double.MIN_VALUE,Double.MIN_VALUE};
        for(double[] num : X_samples) // Get the minimum and maximum per feature
        {
            // Compare feature 1
            if(num[0] < min[0]) min[0] = num[0]; // min
            if(num[0] > max[0]) max[0] = num[0]; // max
            // Compare feature 2
            if(num[1] < min[1]) min[1] = num[1]; // min
            if(num[1] > max[1]) max[1] = num[1]; // max
        }

        XYSeries series1 = new XYSeries("Boundary 1");
        XYSeries series2 = new XYSeries("Boundary 2");
        XYSeries series3 = new XYSeries("Class 1");
        XYSeries series4 = new XYSeries("Class 2");

        double offset= 0.3;
        for(double x = min[0]-offset; x < max[0]+offset; x= x + 0.1)
        {
            for(double y = min[1]-offset; y < max[1]+offset; y= y + 0.1)
            {
                double[][] sample = {{x,y}};
                double[][] prediction = classif.predict(sample);

                // Determine the sample class
                if(prediction[0][0] < 0.5) series1.add(x,y);
                else series2.add(x,y);
            }
        }

        /*
        * Adding sample points
        * */
        for(int i = 0; i < X_samples.length; i++)
        {
            if(y_samples[i][0] == 0.0){
                series3.add(X_samples[i][0],X_samples[i][1]);
            }
            else{
                series4.add(X_samples[i][0],X_samples[i][1]);
            }
        }

        dataset = new XYSeriesCollection();
        dataset.addSeries(series4);
        dataset.addSeries(series3);
        dataset.addSeries(series1);
        dataset.addSeries(series2);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Decision Boundary",
                header[0],
                header[1],
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        // Configure Series
        renderer.setSeriesLinesVisible(0, false);
        renderer.setSeriesShapesVisible(0, true);


        renderer.setSeriesLinesVisible(1, false);
        renderer.setSeriesShapesVisible(1, true);


        renderer.setSeriesLinesVisible(2, false);
        renderer.setSeriesShapesVisible(2, true);
        Color transparentBlue = new Color(0, 0, 255, 100);
        renderer.setSeriesPaint(2, transparentBlue);

        renderer.setSeriesLinesVisible(3, false);
        renderer.setSeriesShapesVisible(3, true);
        Color transparentRed = new Color(255, 0, 0, 100);
        renderer.setSeriesPaint(3, transparentRed);

        renderer.setSeriesLinesVisible(4, false);
        renderer.setSeriesShapesVisible(4, true);

        // Apply configuration
        plot.setRenderer(renderer);

        // Change the grid style
        plot.setBackgroundPaint(Color.darkGray);
        plot.setRangeGridlinePaint(Color.white);
        plot.setDomainGridlinePaint(Color.white);

        // Add chart to the decision boundary.
        if(decisionBoundaryPanel != null)       // Remove old chart
            mainFrame.remove(decisionBoundaryPanel);

        decisionBoundaryPanel = new ChartPanel(chart);
        decisionBoundaryPanel.setMouseWheelEnabled(false);
        decisionBoundaryPanel.setPreferredSize(new Dimension(600, 500));

        return true;
    }

    /***
     * Method to cast an List<List<Double>> into a Double[][] list.
     * @param list List to be cast to a double[][]
     * @return array
     */
    private double[][] convertList2Array(List<List<Double>> list)
    {
        int size = list.size();
        double[][] array = new double[size][2];

        // Casting a List to an Array
        for(int i = 0; i < size; i++)
        {
            List<Double> row = list.get(i);
            if(row == null)
            {
                System.err.println("Error: Row at index " + i);
                continue;
            }
            for(int j = 0; j < row.size(); j++)
                array[i][j] = row.get(j);
        }

        return array;
    }
}
