package it.unipd.dei.bdc1718;

/** Group members:
 *   Roberto Campedelli, Andrea Tommasi, Alberto Forti, Teofan Clipa
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;
import java.util.Iterator;
import org.apache.spark.api.java.function.PairFunction;


public class G25HM2 {

    public static void main(String[] args) throws FileNotFoundException,IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        /**********************************************************************
         1_ Setup Spark and load documents in memory
         **********************************************************************/
        SparkConf configuration =
                new SparkConf(true)
                        .setAppName("Words counter")
                        .setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(configuration);
        JavaRDD<String> docs = sc.textFile(args[0]).cache();
        docs.count();

        /**********************************************************************
         2_ Runs 3 versions of MapReduce word count and returns their individual
            running times, carefully measured: a version that implements the
            Improved Word count 1: we decided to avoid duplicates acting
            directly on the ArrayList.
         **********************************************************************/
        // Let's start to measure time
        long start = System.currentTimeMillis();

        JavaPairRDD<String, Long> wordcounts = docs.
                flatMapToPair((document) -> {           // <-- Map phase
                String[] tokens = document.split(" ");
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                int index;
                boolean found;
                for (String token : tokens) {
                    index = 0;
                    found = false;
                    while(index < pairs.size()){
                        if(pairs.size() > 0 && token.equalsIgnoreCase(pairs.get(index)._1()) && !found){
                                long wordCount = pairs.get(index)._2();
                                pairs.remove(index);
                                pairs.add(new Tuple2<>(token, wordCount +1));
                                found = true;
                        } //necessary to find word duplicates

                        index++;
                    }
                    if(found == false) pairs.add(new Tuple2<>(token, 1L));
                }
                return pairs.iterator();

            })
                    .groupByKey()                       // <-- Reduce phase
                    .mapValues((it) -> {
                        long sum = 0;
                        for (long c : it) {
                            sum += c;
                        }
                        return sum;
                    });

        // End the measure of time
        long end = System.currentTimeMillis();

        System.out.println("Elapsed time for (1) Improved word count 1: " + (end - start) + " ms.");

        /**********************************************************************
         3_ Runs 3 versions of MapReduce word count and returns their individual
            running times, carefully measured: a version that implements the
            Improved Word count 2.
         **********************************************************************/
        Long N = docs.count();

        start = System.currentTimeMillis();

        JavaPairRDD<Long, Tuple2<String, Long >> wordcounts_2_map = docs.
                flatMapToPair((document) -> {           // <-- Map phase (round 1)
                    String[] tokens = document.split(" ");
                    ArrayList<Tuple2<Long, Tuple2<String, Long>>> intermediatePairs = new ArrayList<>();
                    int index;
                    boolean found;
                    for (String token : tokens) {
                        index = 0;
                        found = false;
                        while (index < intermediatePairs.size()) {
                            if (intermediatePairs.size() > 0 && token.equalsIgnoreCase(intermediatePairs.get(index)._2._1) && !found) {
                                long wordCount = intermediatePairs.get(index)._2()._2();
                                long key = intermediatePairs.get(index)._1();
                                intermediatePairs.remove(index);
                                intermediatePairs.add(new Tuple2<>((key), new Tuple2<>(token, wordCount + 1)));
                                found = true;
                            } //checking duplicates as in Improved word count 1

                            index++;
                        }
                        if (found == false)
                            intermediatePairs.add(new Tuple2<>((long) (Math.random() * Math.sqrt(N)), new Tuple2<>(token, 1L)));
                    }
                    return intermediatePairs.iterator();
                });

            // We decided to split Map/reduce into two different JavaPairRDDs for coding readability

            JavaPairRDD< String, Long > wordcounts_2_reduce =
                    wordcounts_2_map.groupByKey()                   // <- Reduce phase (Round 1)
                                .flatMapToPair((x)-> {
                                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                                    Iterator iter = x._2().iterator();
                                    while (iter.hasNext()) {
                                        Tuple2 t = (Tuple2) iter.next(); // take the next Tuple2..
                                        String key = (String) t._1();    //..word
                                        long wordCount = (long) t._2();  //..count

                                        int index = 0;
                                        boolean found = false;
                                        while (index < pairs.size()) {
                                            if (pairs.size() > 0 && key.equalsIgnoreCase(pairs.get(index)._1) && !found) {
                                                long temp_count = pairs.get(index)._2();
                                                pairs.remove(index);
                                                pairs.add(new Tuple2<>(key, temp_count + wordCount));
                                                found = true;
                                            } //checking duplicates as in Improved word count 1

                                            index++;
                                        }
                                        if (found == false) pairs.add(new Tuple2<>(key, wordCount));

                                    }
                                    return pairs.iterator();
                                }).groupByKey()                 // <- Reduce phase (Round 2)
                            // No Map phase (Round 2) since it is an identity
                        .mapValues((x) -> {
                            long sum = 0;
                            for (long c : x) {
                                sum += c; //summing all values with same key(word)
                            }
                            return sum;
                        });

        // End the measure of time
        end = System.currentTimeMillis();

        System.out.println("Elapsed time for (2) Improved word count 2: " + (end - start) + " ms.");

        /**********************************************************************
         4_ Runs 3 versions of MapReduce word count and returns their individual
         running times, carefully measured: a version that uses the reduceByKey method.
         **********************************************************************/
        start = System.currentTimeMillis();

        // docs is the object that read the file with the information
        JavaPairRDD<String, Long> wordCountWithReduceByKey = docs.flatMapToPair((document) -> {
            String[] tokens = document.split(" ");
            ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
            for (String token : tokens) {
                pairs.add(new Tuple2<>(token, 1L));
            }
            return pairs.iterator();
        }).reduceByKey((counter, occurrence) -> (counter + occurrence));

        end = System.currentTimeMillis();

        System.out.println("Elapsed time for (3) word count with reduceByKey: " + (end - start) + " ms");

        /*************************************************************************************
         5_ We now put the dictionary (word, freqWord) into a new dict (freqWord, word) and
            reorder by frequency of words.
         ************************************************************************************/

        // Here we swap (word, freqWord) into a new dict (freqWord, word)
        JavaPairRDD<Long, String> swapped = wordCountWithReduceByKey.mapToPair(new PairFunction<Tuple2<String, Long>, Long, String>() {
            public Tuple2<Long, String> call(Tuple2<String, Long> item) throws Exception {
                return item.swap();
            }
        });

        // Now we have to order it
        JavaPairRDD<Long, String> order = swapped.sortByKey(new ValueComparator(), false);

        /*************************************************************************************
         6_ Asks the user to input an integer k and returns the k most frequent words
            (i.e., those with largest counts), with ties broken arbitrarily.
         ************************************************************************************/

        System.out.println("How many words do you want to stamp in frequency order?");
        Scanner scanner = new Scanner(System.in);
        while (!scanner.hasNextInt()) scanner.next();
        int k = scanner.nextInt();
        System.out.println("You choose to print " + k + " words...");

        for (Tuple2<Long, String> element : order.take(k)) {
            System.out.println("(" + element._2 + ", " + element._1 + ")");
        }

        System.out.println("Press enter to finish");
        System.in.read();
    }



public static class ValueComparator implements Serializable, Comparator<Long> {

    public int compare(Long a, Long b) {
        if (a < b) return -1;
        else if (a > b) return 1;
        return 0;
        }
    }
}
