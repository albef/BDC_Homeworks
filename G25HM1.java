package it.unipd.dei.bdc1718;

/** Group members:
*   Roberto Campedelli, Andrea Tommasi, Alberto Forti, Teofan Clipa
*/


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;

import static jdk.nashorn.internal.objects.NativeMath.min;
import static org.apache.avro.TypeEnum.b;

public class G25HM1 {

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        /**********************************************************************
        1_ Read an input a file dataset.txt of doubles into a JavaRDD dNumbers
         **********************************************************************/

        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s = new Scanner(new File(args[0]));
        while (s.hasNext()) {
            lNumbers.add(Double.parseDouble(s.next()));
        }
        s.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true)
                .setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);

        /**********************************************************************
         2_ Create a new JavaRDD dDiffavgs containing the absolute value of the
            difference between each element of dNumbers and the arithmetic mean
            of all values in dNumbers.
         **********************************************************************/

        // Average of values with reduce method
        double avgs = (dNumbers.reduce((x, y) -> x + y) / dNumbers.count());
        System.out.println("The average of values is: " + avgs);

        // Another arrayList with the absolute differences between values and avgs
        ArrayList<Double> fNumbers = new ArrayList<>();
        for(int i=0; i < lNumbers.size(); i++)
            fNumbers.add(Math.abs(lNumbers.get(i) - avgs));

        // Create parallel collection with latest values found
        JavaRDD<Double> dDiffavgs = sc.parallelize(fNumbers);
        // Alternatively: JavaRDD<Double> dDiffavgs = dNumbers.map((x) -> Math.abs(x - avgs));

        /**********************************************************************
         3_ Compute and print the minimum value in dDiffavgs. Do it in two ways:
                - using the reduce method;
                - using the min method of the JavaRDD
                  class passing to it a comparator
         **********************************************************************/

        // Using the reduce method
        double minVal = dDiffavgs.reduce((accumulator, currentValue) -> accumulator < currentValue ? accumulator : currentValue);
        System.out.println("The min element with Reduce method is: " + minVal);

        // Using the min method of the JavaRDD. The comparator class is defined below.
        double minVal2 = dDiffavgs.min(new ValueComparator());
        System.out.println("The min element with min method of JavaRDD is: " + minVal2);

        /**********************************************************************
         4_ Compute and print another statistics of your choice on the data in
            dNumbers. We have decided to use the following methods to calculate
            the sum of odd and even number separately. We transform every number
            in a key-value pair (rest of the division by 2, value) with the
            "mapToPair"; so we have the even/odd distinction. Then we use the
            "reduceByKey" method and we print directly the two sums.
         **********************************************************************/

        JavaPairRDD<Double, Double> pair = dNumbers.mapToPair((x) -> {
            return new Tuple2<>(x%2, x);
        }).reduceByKey((x,y) -> x+y);
        pair.foreach(x -> {
            if (x._1 == 0.0)
                System.out.println("The sum of even numbers is: " + x._2);
            else
                System.out.println("The sum of odd numbers is: " + x._2);
            });

        /**********************************************************************
         4bis_ Alternatively we computed also the variance of the numbers
               population: we have decided to do this by E[X^2] - E[X]^2,
               using the map/reduce methods.
         **********************************************************************/

        double variance = (dNumbers.map((x) -> x*x).reduce((x, y) -> x + y)) / dNumbers.count()
                          - Math.pow(avgs,2);
        System.out.println("Variance of the numbers is: " + variance);


    }

        /*******************************************************************************
        This class compare doubles and return max or min of the list of doubles compared
        *******************************************************************************/
        public static class ValueComparator implements Serializable, Comparator<Double> {

            public int compare(Double a, Double b) {
                if (a < b) return -1;
                else if (a > b) return 1;
                return 0;
            }

        }
}
