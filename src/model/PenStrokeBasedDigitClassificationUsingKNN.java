package model;

import java.io.*;
import java.util.*;

public class PenStrokeBasedDigitClassificationUsingKNN {
    private static boolean normalize = false;
    private static int K = 1;

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: java NNClassify training_file test_file [k] [-n]");
            System.exit(1);
        }
        String trainFile = args[0];
        String testFile = args[1];
        for (int i = 2; i < args.length; i++) {
            if (args[i].equals("-n")) {
                normalize = true;
            } else {
                try {
                    K = Integer.parseInt(args[i]);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid K value: " + args[i] + " Using default K = 1");
                }
            }
        }

        List<double[]> trainX = new ArrayList<>();
        List<String> trainY = new ArrayList<>();
        List<double[]> testX = new ArrayList<>();
        List<String> testY = new ArrayList<>();

        readData(trainFile, trainX, trainY);
        readData(testFile, testX, testY);

        double[][] trainArr = trainX.toArray(new double[0][]);
        double[][] testArr  = testX.toArray(new double[0][]);

        if (normalize) {
            normalizeFeatures(trainArr, testArr);
        }

        List<String> predictions = new ArrayList<>();
        List<Double> accs = new ArrayList<>();

        Random rand = new Random(0);  // fixed seed for deterministic tie-break

        for (int i = 0; i < testArr.length; i++) {
            String pred = classifyOne(testArr[i], trainArr, trainY, rand);
            predictions.add(pred);
            String trueLabel = testY.get(i);
            double acc = pred.equals(trueLabel) ? 1.0 : 0.0;
            accs.add(acc);
        }

        int correct = 0;
        try (PrintWriter out = new PrintWriter(new FileWriter("..\\resultFile\\pendigits_result.txt"))) {
            out.printf("%-6s %-10s %-10s %-10s%n", "Index", "Predicted", "Actual", "Status");
            for (int i = 0; i < predictions.size(); i++) {
                if (predictions.get(i).equals(testY.get(i))) {
                    correct++;
                }
                out.printf("%-6d %-10s %-10s %-10s%n", i+1, predictions.get(i), testY.get(i), accs.get(i) == 1.0 ? "correct (✓)" : "wrong (✗)" );
            }
        }

        int total = predictions.size();
        int wrong = total - correct;

        double accuracy = (correct * 100.0) / total;
        double errorRate = (wrong * 100.0) / total;

        System.out.println("Total Samples : " + total);
        System.out.println("Correct       : " + correct);
        System.out.println("Wrong         : " + wrong);
        System.out.printf("Accuracy      : %.2f%%\n", accuracy);
        System.out.printf("Error Rate    : %.2f%%\n", errorRate);
        System.out.println("Predictions written to pendigits_result.txt.");
    }

    private static void readData(String fname, List<double[]> X, List<String> Y) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(fname))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\s+");
                double[] feats = new double[parts.length - 1];
                for (int i = 0; i < feats.length; i++) {
                    feats[i] = Double.parseDouble(parts[i]);
                }
                X.add(feats);
                Y.add(parts[parts.length - 1]);
            }
        }
    }

    private static void normalizeFeatures(double[][] train, double[][] test) {
        int d = train[0].length;
        double[] mean = new double[d];
        double[] std  = new double[d];
        int n = train.length;

        // compute mean
        for (int j = 0; j < d; j++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) {
                s += train[i][j];
            }
            mean[j] = s / n;
        }
        // compute std
        for (int j = 0; j < d; j++) {
            double s2 = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = train[i][j] - mean[j];
                s2 += diff * diff;
            }
            std[j] = Math.sqrt(s2 / n);
            if (std[j] == 0.0) std[j] = 1.0;  // avoid divide by zero
        }

        // normalize train
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                train[i][j] = (train[i][j] - mean[j]) / std[j];
            }
        }
        // normalize test
        for (int i = 0; i < test.length; i++) {
            for (int j = 0; j < d; j++) {
                test[i][j] = (test[i][j] - mean[j]) / std[j];
            }
        }
    }

    private static String classifyOne(double[] v, double[][] trainX, List<String> trainY, Random rand) {
        int n = trainX.length;
        double[] distances = new double[n];
        List<Integer> idxs = new ArrayList<>();

        // compute all distances
        for (int i = 0; i < n; i++) {
            double d2 = 0.0;
            for (int j = 0; j < v.length; j++) {
                double diff = v[j] - trainX[i][j];
                d2 += diff * diff;
            }
            distances[i] = d2;
            idxs.add(i);
        }

        // Sort using stored distances
        idxs.sort(Comparator.comparingDouble(i -> distances[i]));

        // pick k nearest
        Map<String, Integer> counts = new HashMap<>();
        for (int t = 0; t < K && t < idxs.size(); t++) {
            String lab = trainY.get(idxs.get(t));
            counts.put(lab, counts.getOrDefault(lab, 0) + 1);
        }
        // find majority count
        int max = Collections.max(counts.values());
        List<String> tied = new ArrayList<>();
        for (String lab : counts.keySet()) {
            if (counts.get(lab) == max) tied.add(lab);
        }
        // tie-breaking
        String predicted;
        if (tied.size() == 1) {
            predicted = tied.get(0);
        } else {
            predicted = tied.get(rand.nextInt(tied.size()));
        }
        return predicted;
    }
}
