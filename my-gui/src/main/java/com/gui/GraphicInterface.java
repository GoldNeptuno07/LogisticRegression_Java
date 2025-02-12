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
    final private JFrame mainFrame;
    final private JMenuBar  menuBar;
    final private JLabel instructionLabel;
    private ChartPanel decisionBoundaryPanel;

    private String datasetURL;
    final private int numberFeatures = 2;

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

        /* Define Menu Bar Buttons */
        menuBar = new JMenuBar();
        JMenu fitModel = new JMenu("Fit Model");
        JMenu helpWindow = new JMenu("Help");

        JMenuItem selectDataBttn = new JMenuItem("Select Dataset");

        fitModel.add(selectDataBttn);
        menuBar.add(fitModel);
        menuBar.add(helpWindow);

        /* Menu Bar Click Listener (Commands) */
        selectDataBttn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser chooserWindow = new JFileChooser();
                int returnValue = chooserWindow.showSaveDialog(null);

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
        instructionLabel = new JLabel("Please select a dataset to train the Logistic Regression model...");

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
        mainFrame.remove(instructionLabel);

        /* Build Decision Boundary Chart */
        displayDecisionBoundary();

        /* SOUTH Prediction Panel */
        JPanel predictionPanel = new JPanel();
        predictionPanel.setLayout(new FlowLayout());

        JLabel predLabel = new JLabel("Make a Prediction. ");
        JTextField[] featuresFields = new JTextField[numberFeatures];

        predictionPanel.add(predLabel);
        for(int i = 0; i < numberFeatures; i++)
        {
            featuresFields[i] = new JTextField(10);
            predictionPanel.add(featuresFields[i]);
        }

        /* Add widgets to the main frame */
        mainFrame.add(decisionBoundaryPanel);
        mainFrame.add(predictionPanel, BorderLayout.SOUTH);
        mainFrame.pack();
    }

    /**
     * Method to build the decision boundary chart
     * */
    private void displayDecisionBoundary() throws Exception {
        /*
        *   Execute the PipeLine
        * */
        System.out.println("Executing Pipeline...");
        PipeLine pipeline = new PipeLine();
        PipeLine.extractAndTransform(datasetURL, true);

        System.out.println("After Pipeline...");

        Map<String, List<List<Double>>> data_map = pipeline.loadDataset();
        String[] header = pipeline.get_header();

        System.out.println("After Load Data...");

        // Get X and y set
        double[][] X_samples = convertList2Array(data_map.get("X"));
        double[][] y_samples = convertList2Array(data_map.get("y"));

        for(int i = 0; i < X_samples.length; i++)
            System.out.println(y_samples[i][0]);

        /*
        * Train Model
        * */
        System.out.println("Training the model...");
        LogisticRegression classif = new LogisticRegression(2, 0.01);
        classif.fit(X_samples, y_samples, 100);

        /*
        * Predicting Grid
        * */
        System.out.println("predicting Grid...");
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

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series1);
        dataset.addSeries(series2);
        dataset.addSeries(series3);
        dataset.addSeries(series4);


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

        renderer.setSeriesLinesVisible(3, false);
        renderer.setSeriesShapesVisible(3, true);

        plot.setRenderer(renderer);

        plot.setBackgroundPaint(Color.darkGray);
        plot.setRangeGridlinePaint(Color.white);
        plot.setDomainGridlinePaint(Color.white);

        decisionBoundaryPanel = new ChartPanel(chart);
        decisionBoundaryPanel.setMouseWheelEnabled(false);
        decisionBoundaryPanel.setPreferredSize(new Dimension(600, 500));
    }

    private double[][] convertList2Array(List<List<Double>> list)
    {
        int size = list.size();
        double[][] array = new double[size][2];

        // Cast List to Array
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
