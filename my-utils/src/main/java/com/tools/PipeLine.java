package com.tools;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.*;
import org.apache.beam.sdk.values.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Pipeline class to performe ETL based on the given dataset. The dataset will be normalized
 * and finally saved into a csv file.
 *
 * @author Angel Cruz
 * */
public class PipeLine {
    /*
     * Path where the standardized data will be store
     * */
    final static String saveInPath = "my-utils/src/main/resources/StandardizedData";

    /* Array to store the dataset header */
    public String[] header;

    /**
     * Main PipeLine constructor, we will save the dataset path to carry out the
     * normalization of the data.
     * */
    public PipeLine()
    {}

    public static void main(String[] args)
    {
        PipeLine pipeline = new PipeLine();
        PipeLine.extractAndTransform("/Users/angel_c/Downloads/data.csv", true);

    }

    /**
     * Method to carry out the extraction and transformation of the data, and finally return
     * the normalized data.
     *
     * @param dataPath Dataset file path
     * @param hasHeader If the dataset has headers, otherwise add default headers
     * */
    public static void extractAndTransform(String dataPath, boolean hasHeader)
    {
        /* Define the pipeline */
        PipelineOptions options = PipelineOptionsFactory.create();
        Pipeline p = Pipeline.create(options);

        /*
        * Load the dataset from the file. Then, extract the headers to map each
        * value to its corresponding feature.
        */
        PCollection<String> raw_data = p
                    .apply("Load each line of the dataset", TextIO.read().from(dataPath));

        List<String> headers = new ArrayList<>();
        try
        {
            BufferedReader br = new BufferedReader(new FileReader(dataPath));
            String headerLine = br.readLine();
            headers = Arrays.asList(headerLine.split(","));
        }
        catch(Exception e)
        {
            System.out.println("Exception caught. " + e.getMessage());
        }

        PCollection<String> headersPCollection = p
                .apply("CreateHeaders", Create.of(String.join(",", headers)));
        PCollectionView<List<String>> headersView = p
                .apply("Transform to PCollectionView", Create.of(Collections.singletonList(headers)))
                .apply(View.asSingleton());

        /*
         * Since the Linear Regression model take two numerical features as input, we'll
         * get the first two numerical features that we find.
         */
        PCollection<String> rows = raw_data.apply(ParDo.of(new DoFn<String, String>() {
            public static boolean isNumeric(String strNum)
            {
                if (strNum == null)
                    return false;
                try
                {
                    double d = Double.parseDouble(strNum);
                }
                catch (NumberFormatException nfe)
                {
                    return false;
                }
                return true;
            }

            @ProcessElement
            public void apply(@Element String row, ProcessContext c)
            {
                String[] values = row.split(",");
                int n = values.length;
                List<String> num_features = new ArrayList<String>();

                try
                {
                    for(String val : values)
                    {
                        if (isNumeric(val))
                            num_features.add(val);

                        if (num_features.size() == 2)
                            break;
                    }

                    if (num_features.size() != 2)
                        return;

                    num_features.add(values[n-1]);  // Add the Target
                    c.output(String.join(",", num_features));
                }
                catch (Exception e){}
            }
        }));

        /*
        * Since "population" PCollection will be an unordered KV map, we could lose the order of the
        * features. We can use this structure to compute statistics, like mean and standard deviation
        * efficiently using Mean.perKey().
        *
        * And we'll use "samples" PCollection to Standardize the data, since, it preserves the order of
        * corresponding features in a single List.
        *
        * */
        final TupleTag<KV<String, Double>> populationTag = new TupleTag<KV<String, Double>>(){};
        final TupleTag<List<KV<String, Double>>> samplesTag = new TupleTag<List<KV<String, Double>>>(){};

        PCollectionTuple mixedCollection = rows
                .apply("Map values into KV<Feature,Value> | List<KV<Feature,Value>>",
                        ParDo.of(new DoFn<String, KV<String, Double>>() {
                            @ProcessElement
                            public void apply(@Element String line, ProcessContext c)
                            {
                                List<String> headers = c.sideInput(headersView);
                                String[] values = line.split(",");
                                int n = headers.size();
                                String key;
                                Double value;
                                try
                                {
                                    List<KV<String,Double>> sample = new ArrayList<>();
                                    for(int i = 0; i < n - 1; i++)
                                    {
                                        key = headers.get(i);
                                        value = Double.parseDouble(values[i]);

                                        // population
                                        c.output(KV.of(key,value));
                                        // samples
                                        sample.add(KV.of(key,value));
                                    }
                                    // Add the target
                                    sample.add(KV.of("Target", Double.parseDouble(values[n-1])));
                                    c.output(samplesTag, sample);
                                }
                                catch(Exception e){}
                            }
                        }).withSideInputs(headersView).withOutputTags(populationTag, TupleTagList.of(samplesTag)));

        PCollection<KV<String,Double>> population = mixedCollection.get(populationTag);
        PCollection<List<KV<String,Double>>> samples = mixedCollection.get(samplesTag);

        /*
        * In order to standardize the data, first we need to compute the necessary values, such as,
        * the mean and standard deviation. Once we have calculated those values, we will standardize
        * the data and finally save it in a csv file.
        *
        * */

        /*
        * Computing the population's mean
        *
        */
        PCollection<KV<String, Double>> meanPerKey = population
                .apply("Compute MeanPerKey", Mean.perKey());
        PCollectionView<List<KV<String,Double>>> meanView = meanPerKey
                .apply(View.asList());

        /*
        * Computing the population's standard deviation
        *
        */
        PCollection<KV<String,Double>> stdPerKey = population
                .apply(ParDo.of(new DoFn<KV<String, Double>, KV<String, Double>>() {
                    @ProcessElement
                    public void squaredDifference(@Element KV<String,Double> e, ProcessContext c)
                    {
                        List<KV<String,Double>> meanList = c.sideInput(meanView);
                        Double result;
                        String key;

                        for(KV<String,Double> mean : meanList)
                        {
                            if(mean.getKey().equals(e.getKey()))
                            {
                                key = e.getKey();
                                result = Math.pow(e.getValue() - mean.getValue(), 2);

                                // Return the SquaredDifference
                                c.output(KV.of(key,result));
                                break;
                            }
                        }
                    }
                }).withSideInputs(meanView))
                .apply(Mean.perKey())
                .apply("Squared Root", ParDo.of(new DoFn<KV<String, Double>, KV<String, Double>>() {
                    @ProcessElement
                    public void squaredRoot(@Element KV<String,Double> e, ProcessContext c)
                    {
                        String key = e.getKey();
                        Double value = Math.sqrt(e.getValue());

                        c.output(KV.of(key,value));
                    }
                }));
        PCollectionView<List<KV<String,Double>>> stdView = stdPerKey
                .apply(View.asList());

        /*
        * Once we have calculated the necessary statistics, we can start standardizing the samples,
        * as in the following code.
        *
        * */
        PCollection<List<KV<String,Double>>> standardizedData = samples
                .apply("Standardizing Data", ParDo.of(new DoFn<List<KV<String,Double>>, List<KV<String,Double>>>() {
                    @ProcessElement
                    public void standardizeData(@Element List<KV<String, Double>> e, ProcessContext c)
                    {
                        String key;
                        Double value;
                        List<KV<String,Double>> output = new ArrayList<>();
                        Double mean = 0.0, std = 1.0;
                        List<KV<String,Double>> meanList = c.sideInput(meanView);
                        List<KV<String,Double>> stdList = c.sideInput(stdView);

                        for(KV<String,Double> v : e)
                        {
                            if(v.getKey().equals("Target"))
                            {
                                output.add(KV.of("Target", v.getValue()));
                                continue;
                            }

                            if (meanList.get(0).getKey().equals(v.getKey()))
                                mean = meanList.get(0).getValue();
                            else
                                mean = meanList.get(1).getValue();

                            if (stdList.get(0).getKey().equals(v.getKey()))
                                std = stdList.get(0).getValue();
                            else
                                std = stdList.get(1).getValue();

                            key = v.getKey();
                            value = (v.getValue() - mean) / std; /* Standardize value */

                            output.add(KV.of(key, value));
                        }
                        c.output(output);
                    }
                }).withSideInputs(meanView, stdView));


        /*
        * Finally, we can save the final samples, which are the one we already preprocess
        *
        * */
        PCollection<String> PreprocessedData = standardizedData.apply("Turn sample to a string line",
                ParDo.of(new DoFn<List<KV<String, Double>>, String>() {
            @ProcessElement
            public void apply(@Element List<KV<String, Double>> e, ProcessContext c)
            {
                List<String> headers = c.sideInput(headersView);
                String output, header;

                header = headers.get(0);
                if (e.get(0).getKey().equals(header))
                    output = e.get(0).getValue() + "," + e.get(1).getValue();
                else
                    output = e.get(1).getValue() + "," + e.get(0).getValue();

                // Add target
                output = output + "," + e.get(2).getValue();

                c.output(output);
            }
        }).withSideInputs(headersView));

        // Concatenate heades and standardized samples
        PCollectionList<String> combinedElements = PCollectionList.of(headersPCollection).and(PreprocessedData);
        PCollection<String> finalDataset = combinedElements.apply("Combine Header and Data", Flatten.pCollections());

        // Save the final data into a csv file
        finalDataset
                .apply("Save Standardized Data into a csv file", TextIO.write().to(saveInPath).withNumShards(1).withSuffix(".csv"));

        try
        {
            p.run().waitUntilFinish();
        }
        catch (Exception e)
        {
            System.out.println("An error has ocurred.\n Exception caught. " + e.getMessage());
        }
    }

    public Map<String,List<List<Double>>> loadDataset()
    {
        // Define csv file url
        String url = saveInPath + "-00000-of-00001.csv";

        // Initialize arrays to store the data
        List<List<Double>> x_data = new ArrayList<>();
        List<List<Double>> y_data = new ArrayList<>();

        try
        {
            String line, token;
            StringTokenizer tokens;
            BufferedReader reader = new BufferedReader(new FileReader(url));

            // Save header
            this.header = reader.readLine().split(",");

            // Read the file lines
            while((line = reader.readLine()) != null)
            {
                tokens = new StringTokenizer(line, ",");
                List<Double> sample = new ArrayList<>();
                while(tokens.hasMoreTokens())
                {
                    token = tokens.nextToken();
                    double finalToken = Double.parseDouble(token);

                    if(sample.size() == 2) // Save sample's label
                    {
                        y_data.add(new ArrayList<Double>(){{
                            add(finalToken);
                        }});
                        break;
                    }
                    // Store numerical value
                    sample.add(finalToken);
                }
                // Store sample
                x_data.add(sample);
            }
        }
        catch (FileNotFoundException e)
        {
            System.out.println("File Not Found, exception caught. " + e.getMessage());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Store data into a Hash Map
        Map<String, List<List<Double>>> data = new HashMap<String, List<List<Double>>> ();

        data.put("X", x_data);
        data.put("y", y_data);

        return data;
    }

    public String[] get_header()
    {
        return this.header;
    }
}



