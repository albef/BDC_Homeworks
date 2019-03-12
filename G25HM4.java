package it.unipd.dei.bdc1718;

/**
 * Big Data Computing - Fourth Homework
 *
 * @author  Roberto Campedelli, Andrea Tommasi, Alberto Forti, Teofan Clipa
 * @version 1.0
 * @since   2018-05-30
 */

import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import org.apache.spark.SparkConf;

import java.util.Scanner;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;


public class G25HM4 {
    /**
     * runMapReduce provide a solution for diversity maximization problem.
     * To do this it follows 4 steps:
     *      (a) partitions the input set of points into numBlocks subset;
     *      (b) extract k points from each subset by running Farthest-First Traversal;
     *      (c) gather numBlocks * k points into a coreset vector;
     *      (d) return a set of k points determined by sequential max-diversity alg.
     * @param pointsrdd set of points
     * @param k number of points to be extracted
     * @param numBlocks number of partitions
     * @return k distinct points exploiting the max-diversity algorithm
     */
    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsrdd,int k, int numBlocks) throws FileNotFoundException {
        long startTimeCoreset;
        long endTimeCoreset;

        long startTimeFinal;
        long endTimeFinal;

        startTimeCoreset = System.currentTimeMillis();
        PrintWriter writer = new PrintWriter("times.txt");

        // Create the partitions (coalesce if shuffle necessary)
        JavaRDD<Vector> repartitioned = pointsrdd.repartition(numBlocks);

        /**
         *      We define a new RDD with points extracted with kcenter function
         *      using mapPartitions (which uses an iterator), creating an ArrayList
         *      provided in input to kcenter, then returning an iterator related to
         *      the extracted points
         */

        JavaRDD<Vector> kcenterset = repartitioned.mapPartitions((iter)->{
            ArrayList<Vector> partitionedpoints = new ArrayList<>();
            while(iter.hasNext()){
                partitionedpoints.add(iter.next()); }
            ArrayList<Vector> kcenteredpoints = kcenter(partitionedpoints,k);
            return kcenteredpoints.iterator();
            }
        );

        // We need to gather the coreset vector now, using the ArrayList constructor
        ArrayList<Vector> coreset = new ArrayList<>(kcenterset.collect());

        endTimeCoreset = System.currentTimeMillis();

        System.out.println("Time taken by the coreset construction is " +
                (endTimeCoreset-startTimeCoreset) + " ms.");
        writer.println("Time taken by the coreset construction is " +
                (endTimeCoreset-startTimeCoreset) + " ms.");

        // Finally, we can run the Sequential method
        startTimeFinal = System.currentTimeMillis();
        ArrayList<Vector> result = runSequential(coreset,k);
        endTimeFinal = System.currentTimeMillis();

        System.out.println("Time taken by the computation of final solution is "
                + (endTimeFinal-startTimeFinal) + " ms.");
        writer.println("Time taken by the computation of final solution is "
                + (endTimeFinal-startTimeFinal) + " ms.");
        writer.close();
        return result;

    }

    /**
     * Computes the average distance between points in ArrayList of points
     * @param pointslist set of points
     * @return average distance between all points in pointslist
     * @throws NullPointerException if passed a null reference
     */
    public static double measure(ArrayList<Vector> pointslist){
        if (pointslist == null)
            throw new NullPointerException();

        int numPoints = pointslist.size();

        // If the array is empty, returns 0 as average distance
        if (numPoints == 0)
            return 0;

        double sumOfDistances = 0.;
        int numberOfComputedDistances = 0;

        // Brute force approach, computes all the distances
        for (int i = 0; i < numPoints; i++)
        {
            Vector v = pointslist.get(i);
            for (int j = 0; j < numPoints; j++)
            {
                if (j != i)
                {
                    sumOfDistances += Math.sqrt(Vectors.sqdist(v, pointslist.get(j)));
                    numberOfComputedDistances++;
                }
            }
        }

        return sumOfDistances / numberOfComputedDistances;
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        /**
         * 1_ Setting up Spark and load data/input parameters
         */
        SparkConf configuration =
                new SparkConf(true)
                        .setAppName("Diversity maximization");
                        //.setMaster("local[*]"); // uncomment if use locally

        // We also provide a file with the results
        PrintWriter writer = new PrintWriter("distancepoints.txt");

        JavaSparkContext sc = new JavaSparkContext(configuration);

        Scanner in = new Scanner(System.in);
        int numBlocks;
        int k;

        System.out.print("Number of partitions: ");
        numBlocks = in.nextInt();
        System.out.println();

        if(numBlocks < 1)
            throw new IOException("Number of partitions must be greater than 1");

        System.out.print("Number of extracted points: ");
        k = in.nextInt();
        System.out.println();

        if(k < 0)
            throw new IOException("k must be greater than 0");

        // Reads the input file and store in memory
        JavaRDD<Vector> data = sc.textFile(args[0]).map(InputOutput::strToVector).cache();

        /**
         * 2_ Computing solution for Max-diversity problem
         */
        ArrayList<Vector> result = runMapReduce(data,k,numBlocks);

        System.out.println("The average distance among the final result points is: "
                + measure(result));
        writer.println("The average distance among the final result points is: "
                + measure(result));

        System.out.println("Distinct points found by diversity maximization algorithm are: ");
        writer.println("Distinct points found by diversity maximization algorithm are: ");
        for(Vector points : result){
            System.out.println(points);
            writer.println(points);
        }

        writer.close();


    } // Main closed



    /**
     * Sequential approximation algorithm based on matching, already provided
     * @param points set of points
     * @param k numbers of points to be extracted
     * @return set of k points
     */
    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {
        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter = 0; iter < k / 2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i + 1; j < n; j++) {
                        if (candidates[j]) {
                            double d = Math.sqrt(Vectors.sqdist(points.get(i), points.get(j)));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;
    }

    /**
     * Computed a set of centers using the Farthest-First Traversal algorithm
     * ----> Imported from previous homework <----
     * @param P ArrayList<Vector> which contain all the points of the dataset
     * @param k integer that represents the number of center clusters you want to have
     * @return ArrayList<Vector> which contain the center computed with the kcenter algorithm
     * @throws IllegalArgumentException if k is greater than the number of points
     */
    public static ArrayList<Vector> kcenter(ArrayList<Vector> P, int k) throws IllegalArgumentException
    {

        int PSize = P.size();
        // A control on the input
        if (k > PSize)
            throw new IllegalArgumentException("K cannot be greater than the number of points!");

        // Choose a random point as first center
        final boolean chosen [] = new boolean[PSize];
        final int firstPointIndex = (int)(Math.random() * PSize);

        // Set to true that we chosen the point
        chosen[firstPointIndex] = true;

        // Add the point to the centers
        ArrayList<Vector> centers = new ArrayList<>();
        centers.add(P.get(firstPointIndex));

        // Let's create the matrix for the distances
        double [] dist = new double[PSize];

        // Let's memorize the minDistances
        double minDist[] = new double[PSize];

        // Now we compute the distances from all the points to the center
        int index = 0;
        for (Vector v : P)
        {
            dist [index] = Vectors.sqdist(v, centers.get(0));
            minDist[index] = dist [index++];
        }

        // Let's find all the other centers
        for (int h = 1; h < k; h++)
        {
            double max = -1;
            index = -1;

            // First, compute the distances from the new center, then find the max min
            for (int j = 0; j < PSize; j++)
            {
                if (chosen[j] == false)
                {
                    // Compute all the new distances
                    dist[j] = Vectors.sqdist(P.get(j), centers.get(h-1));
                    if (dist[j] < minDist[j])
                        minDist[j] = dist[j];
                }
            }

            // We choose the maximum minimum value
            for (int j = 0; j < PSize; j++)
            {
                if (chosen[j] == false) {
                    if (minDist[j] > max) {
                        max = minDist[j];
                        index = j;
                    }
                }
            }

            // Add the new center
            centers.add(P.get(index));
            chosen[index] = true;
        }
        return centers;
    }


}
