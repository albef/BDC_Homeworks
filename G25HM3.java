package it.unipd.dei.bdc1718;

/**
 * Big Data Computing - Third Homework
 *
 * @author  Roberto Campedelli, Andrea Tommasi, Alberto Forti, Teofan Clipa
 * @version 1.0
 * @since   2018-05-08
 */

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G25HM3 {

    /**
     * Computed a set of centers using the Farthest-First Traversal algorithm
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

    /**
     * Computes with a weighted variant of the kmeans++ algorithm, where, in each iteration, the probability for a
     * non-center point p of being chosen as next center is
     *
     * w_p*(d_p)^2/(sum_{q non center} w_q*(d_q)^2)
     *
     * where d_p is the distance of p from the closest among the already selected centers and w_p is the weight of p.
     *
     * @param P ArrayList<Vector> which contain all the points of the dataset
     * @param WP ArrayList<Long> of weights for the points in P
     * @param k integer that represents the number of center clusters you want to have
     * @return ArrayList<Vector> which contain the center computed with the kmeans++ algorithm
     * @throws IllegalArgumentException if k is greater than the number of points
     */
    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> WP, int k) throws IllegalArgumentException
    {
        // Find the number of points
        final int numPoints = P.size();

        // A control on the input
        if (k > numPoints)
            throw new IllegalArgumentException("K cannot be greater than the number of points!");

        if (WP.size() != numPoints)
            throw new IllegalArgumentException("WP must have the same dimension of P");

        // Set the corresponding element in this array to indicate when elements of P are no longer available
        final boolean[] chosen = new boolean[numPoints];

        // Set the returned ArrayList
        ArrayList<Vector> centers = new ArrayList<>();

        // Choose the first center randomly
        final int firstPointIndex = (int)(Math.random() * numPoints);
        centers.add(P.get(firstPointIndex));
        chosen [firstPointIndex] = true;

        // We use an array to keep track of the squared distances
        final double [] minDistSquared = new double[numPoints];

        // Initialize the distances from the first center, weighted:
        for (int i = 0; i < numPoints; i++) {
            if (i != firstPointIndex) { // That point isn't considered
                double d = Vectors.sqdist(P.get(firstPointIndex), P.get(i));
                minDistSquared[i] = d*d*(WP.get(i));
            }
        }

        // We define a vector for cumulative distribution function
        ArrayList<Double> cumulative_probability;

        while (centers.size() < k) {

            // We sum up the squared distances for the points in P not already taken
            double distSqSum = 0.0;

            // We compute sum_qnoncenter(w_q * (d_q)^2)
            for (int i = 0; i < numPoints; i++) {
                if (!chosen[i]) {
                    distSqSum += minDistSquared[i];
                }
            }

            cumulative_probability = new ArrayList<>();
            double probability_sum = 0;

            // Compute the CDF
            for(int i = 0; i < numPoints; i++){
                probability_sum += (minDistSquared[i] / distSqSum);
                cumulative_probability.add(probability_sum);
            }

            // We take a random value and we use it for extract the next center
            double r = Math.random();
            // Index of the next center point
            int nextPointIndex = 0;

            for(int j = 0; j < cumulative_probability.size(); j++){
                // if r is less than the j-th cumulative probability,
                // then we take the j-th point as next center, since CDF
                // partitions the probability space
                if(r < cumulative_probability.get(j)) {
                    nextPointIndex = j;
                    break;
                }
            }

            centers.add(P.get(nextPointIndex));
            // Mark it as taken.
            chosen[nextPointIndex] = true;

            if (centers.size() < k) {
                // Now update elements of minDistSquared.  We only have to compute
                // the distance to the new center to do this.
                for (int j = 0; j < numPoints; j++) {
                    // Only have to worry about the points still not taken.
                    if (!chosen[j]) {
                        // Updating the minimum squared distance considering the new center
                        double d = Vectors.sqdist(P.get(j),centers.get(centers.size()-1));
                        double d2 = d * d * WP.get(j);
                        if (d2 < minDistSquared[j]) {
                            minDistSquared[j] = d2;
                        }
                    }
                }
            }

        }
        return centers;
    }

    /**
     *
     * @param P ArrayList<Vector> which contains the points of the dataset
     * @param C ArrayList<Vector> which contains the centers of the clusters of the dataset
     * @return the average squared distance of a point of P from its closest center
     */
    public static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C){
        double averageSquaredDistance = 0.;

        int PSize = P.size();
        int CSize = C.size();
        double minDist [] = new double[PSize];

        for (int j = 0; j < CSize; j++) {
            for (int i = 0; i < PSize; i++) {
                // The first run computes all distances(if the point is not a center)
                // since minDist is empty, then it compares the distances and stores the minimum
                if ((j == 0 || minDist[i] > Vectors.sqdist(P.get(i), C.get(j))) && !C.contains(P.get(i)))
                    minDist[i] = Vectors.sqdist(P.get(i), C.get(j));
            }
        }

        for (int i = 0; i < PSize; i++)
            averageSquaredDistance += minDist[i];

        return averageSquaredDistance / (PSize - CSize);
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        Scanner in = new Scanner(System.in);
        int k, k1, dataset_choice, dimensions;

        System.out.print("k value: ");
        k = in.nextInt();
        System.out.println();

        System.out.print("k1 value(must be greater than k): ");
        k1 = in.nextInt();
        System.out.println();

        if(k1 < k)
            throw new IOException("k1 cannot be lower than k");

        System.out.println("1: 10k points, 2: 50k points, 3: 100k points, 4: 500k points");
        dataset_choice = in.nextInt();
        System.out.println();

        // Let the user select the dataset
        ArrayList<Vector> selected;
        switch(dataset_choice) {
            case 1:
                selected =  InputOutput.readVectorsSeq(args[0]);
                break;
            case 2:
                selected =  InputOutput.readVectorsSeq(args[1]);
                break;

            case 3:
                selected =  InputOutput.readVectorsSeq(args[2]);
                break;

            case 4:
                selected =  InputOutput.readVectorsSeq(args[3]);
                break;

            default:
                selected =  InputOutput.readVectorsSeq(args[0]);
                break;
        }

        dimensions = selected.size();

        /***********************************************************
         *  1_  Runs kcenter and compute its running time
         ***********************************************************/
        long start = System.currentTimeMillis();
        ArrayList<Vector> centers_1 = kcenter(selected,k);
        long end = System.currentTimeMillis();
        System.out.println("Elapsed time " + (end - start) + " ms");


        /***********************************************************
         *  2_  Runs kmeans++ with all weights equal to 1, and then
         *      runs kmeansObj to obtain the average square distance
         ***********************************************************/
        ArrayList<Long> weights = fillVector(dimensions,false);
        ArrayList<Vector> centers_2 = kmeansPP(selected, weights, k);
        System.out.println("Average squared distance of a point of P from its closest center: " +
                kmeansObj(selected,centers_2));

        /***********************************************************
         *  3_  Runs kcenter(P,k1) to obtain a set of k1 centers X;
         *  then runs kmeansPP(X,WX,k) to obtain a set of k centers
         *  C, and finally runs kmeansObj(P,C).
         ***********************************************************/
        ArrayList<Vector> X = kcenter(selected,k1);
        ArrayList<Long> weights_X = fillVector(X.size(),false); /// <- Initially all weights equals to 1
        ArrayList<Vector> C = kmeansPP(X,weights_X,k);
        System.out.println("Average squared distance of a point of P from its closest" +
                " center, after making a subset of k over k1 centers: " +
                kmeansObj(selected,C));

    }

    /**
     * Method which returns an array filled by ones or random values
     * @param dimensions size of the array
     * @return vector filled
     */
    private static ArrayList<Long> fillVector(int dimensions, boolean random) {
        long filler = 0;
        ArrayList<Long> vector = new ArrayList<>();
        for(int i=0;i<dimensions;i++){
            if(!random)
                vector.add((long)1);
            else{
                filler = (long)(Math.random()*1000);
                vector.add(filler);
            }
        }
        return vector;
    }
}
