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

/**
 * This class will be used to generate the main GUI, in order to define the layout of the widgets and
 * show the decision boundary of the Logistic Regression model.
 *
 * @author Angel Cruz
 *
 * */
public class GraphicInterface extends JFrame {
    private JFrame mainFrame;
    private JMenuBar  menuBar;
    private JLabel instructionLabel;
    private ChartPanel decisionBoundaryPanel;

    private String datasetURL;
    private int numberFeatures = 2;

    /**
     * Main constructor to define the main frame and all its widgets
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
                    try
                    {
                        JFileChooser chooserWindow = new JFileChooser();
                        chooserWindow.showSaveDialog(null);
                        datasetURL = chooserWindow.getSelectedFile().getAbsolutePath();

                        displayModelWidgets();
                    }
                    catch (Exception exc)
                    {
                        System.err.println("PATH RECEIVE IS NULL. \n" + e);
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
     *
     * */
    private void displayModelWidgets()
    {
        mainFrame.remove(instructionLabel);

        /* Build Decision Boundary Chart */
        displayDecisionBoundary();

        /* SOUTH Prediciton Panel */
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
     * Method to build the decicion boundary chart
     * */
    private void displayDecisionBoundary()
    {
        XYSeries series1 = new XYSeries("Class 1");
        XYSeries series2 = new XYSeries("Class 2");
        for(double y = 0; y <= 2; y += 0.1)
        {
            for(double x = 0; x <= 5; x += 0.1)
            {
                series1.add(x, y);
                series2.add(x, y + 2);
            }
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series1);
        dataset.addSeries(series2);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Decision Boundary",
                "Feature 1",
                "Feature 2",
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

        plot.setRenderer(renderer);

        plot.setBackgroundPaint(Color.darkGray);
        plot.setRangeGridlinePaint(Color.white);
        plot.setDomainGridlinePaint(Color.white);

        decisionBoundaryPanel = new ChartPanel(chart);
        decisionBoundaryPanel.setMouseWheelEnabled(false);
        decisionBoundaryPanel.setPreferredSize(new Dimension(600, 500));
    }

    public static void main(String[] args)
    {
        GraphicInterface gui = new GraphicInterface();
    }
}
