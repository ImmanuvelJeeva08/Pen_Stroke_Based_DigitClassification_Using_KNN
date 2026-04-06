package model;

import java.io.*;
import java.util.*;

public class PenStrokeBasedDigitClassificationUsingKNN {
    private static boolean normalize = false;
    private static int bestK = 1;

    public static void main(String[] args) throws IOException {

        String trainFile = "src/trainingFile/pendigits_training.txt";
        String testFile  = "src/testFile/pendigits_test.txt";

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

        // 🔥 Find best K using validation
        bestK = findBestK(trainArr, trainY);
        System.out.println("Best K selected: " + bestK);

        List<String> predictions = new ArrayList<>();
        List<Double> accs = new ArrayList<>();

        Random rand = new Random(0);

        for (int i = 0; i < testArr.length; i++) {
            String pred = classifyOne(testArr[i], trainArr, trainY, rand, bestK);
            predictions.add(pred);

            String trueLabel = testY.get(i);
            double acc = pred.equals(trueLabel) ? 1.0 : 0.0;
            accs.add(acc);
        }

        try (PrintWriter out = new PrintWriter(new FileWriter("src/resultFile/pendigits_result.txt"))) {
            out.println("Index Predicted Actual Correct"); // header added
            for (int i = 0; i < predictions.size(); i++) {
                out.printf("%d %s %s %.0f%n", i, predictions.get(i), testY.get(i), accs.get(i));
            }
        }

        double sum = 0.0;
        for (double a : accs) sum += a;
        double overall = sum / accs.size();

        System.out.printf("Overall accuracy: %.6f%n", overall);
    }

    // ================= READ DATA =================
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

    // ================= NORMALIZATION =================
    private static void normalizeFeatures(double[][] train, double[][] test) {
        int d = train[0].length;
        double[] mean = new double[d];
        double[] std  = new double[d];

        int n = train.length;

        // mean
        for (int j = 0; j < d; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += train[i][j];
            mean[j] = sum / n;
        }

        // std
        for (int j = 0; j < d; j++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                double diff = train[i][j] - mean[j];
                sum += diff * diff;
            }
            std[j] = Math.sqrt(sum / n);
            if (std[j] == 0) std[j] = 1;
        }

        // normalize train
        for (int i = 0; i < train.length; i++) {
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

    // ================= FIND BEST K =================
    private static int findBestK(double[][] trainX, List<String> trainY) {

        int split = (int)(trainX.length * 0.8);

        double[][] trainPart = Arrays.copyOfRange(trainX, 0, split);
        List<String> trainYPart = trainY.subList(0, split);

        double[][] valX = Arrays.copyOfRange(trainX, split, trainX.length);
        List<String> valY = trainY.subList(split, trainY.size());

        int[] kValues = {1, 3, 5, 7};
        int bestK = 1;
        double bestAcc = 0;

        Random rand = new Random(0);

        for (int k : kValues) {

            int correct = 0;

            for (int i = 0; i < valX.length; i++) {
                String pred = classifyOne(valX[i], trainPart, trainYPart, rand, k);
                if (pred.equals(valY.get(i))) correct++;
            }

            double acc = (double) correct / valX.length;
            System.out.println("K=" + k + " Validation Accuracy=" + acc);

            if (acc > bestAcc) {
                bestAcc = acc;
                bestK = k;
            }
        }

        return bestK;
    }

    // ================= CLASSIFY =================
    private static String classifyOne(double[] v, double[][] trainX, List<String> trainY, Random rand, int K) {

        int n = trainX.length;
        double[] distances = new double[n];
        List<Integer> idxs = new ArrayList<>();

        // compute distances
        for (int i = 0; i < n; i++) {
            double d2 = 0;
            for (int j = 0; j < v.length; j++) {
                double diff = v[j] - trainX[i][j];
                d2 += diff * diff;
            }
            distances[i] = d2;
            idxs.add(i);
        }

        // sort by distance
        idxs.sort(Comparator.comparingDouble(i -> distances[i]));

        // pick K neighbors
        Map<String, Integer> counts = new HashMap<>();
        for (int t = 0; t < K && t < idxs.size(); t++) {
            String label = trainY.get(idxs.get(t));
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        // majority vote
        int max = Collections.max(counts.values());
        List<String> tied = new ArrayList<>();

        for (String label : counts.keySet()) {
            if (counts.get(label) == max) {
                tied.add(label);
            }
        }

        // tie-break
        if (tied.size() == 1) {
            return tied.get(0);
        } else {
            return tied.get(rand.nextInt(tied.size()));
        }
    }
}
