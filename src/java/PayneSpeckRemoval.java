package net.telepathix.speckremoval;

import android.util.Pair;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

import static java.lang.Math.sqrt;
import static org.opencv.core.Core.CMP_LT;
import static org.opencv.core.Core.NORM_MINMAX;

public class PayneSpeckRemoval {

    public static class InvalidKernelSizeException extends RuntimeException {
        public InvalidKernelSizeException(int kernelSize) {
            super("Invalid kernel size " + kernelSize);
        }
    }

    //the Payne kernel will "score" high for points in a speck and low for points outside a speck
    public static Mat makePayneKernel(int size) {
        if (size % 2 != 1) {
            throw new InvalidKernelSizeException(size);
        }
        Mat mat = new Mat(size, size, CvType.CV_32F);
        float[] buffer = new float[size * size];
        float totalElements = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float elementVal = (float)sqrt(((size/2.0 - i) * (size/2.0 - i) + (size/2.0 - j) * (size/2.0 - j)));
                buffer[i * size + j] = elementVal;
                totalElements += elementVal;
            }
        }
        mat.put(0, 0, buffer);
        Core.multiply(mat, Mat.ones(mat.size(), mat.type()), mat, 1.0f / totalElements);
        return mat;
    }

    private static final int KERNEL_SIZE = 5;

    public static Pair<Mat,Mat> payneSpeckRemoval(String fileName, int rows, int cols) throws IOException {
        //read the data in a Mat of UInt16
        File file = new File(fileName);
        FileInputStream fileInputStream = new FileInputStream(file);
        FileChannel fileChannel = fileInputStream.getChannel();
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(fileInputStream.available());
        fileChannel.read(byteBuffer);
        Mat rawMat = new Mat(rows, cols, CvType.CV_16U, byteBuffer);

        //make and apply the Payne Kernel
        Mat convolution = new Mat();
        Mat payneKernel = makePayneKernel(KERNEL_SIZE);
        Imgproc.filter2D(rawMat, convolution, CvType.CV_16U, payneKernel);

        //make 2 masks - one will contain points in a speck the other points not in a speck
        Mat mask = new Mat();
        Core.compare(convolution, rawMat, mask, CMP_LT);
        Mat invMask = new Mat();
        Core.bitwise_not(mask, invMask);

        //apply the mask to get points not in a speck
        Mat invMaskedImage = new Mat();
        Core.bitwise_or(rawMat, rawMat, invMaskedImage, invMask);

        //replace the points in a speck by averaging the surrounding points
        Mat invMaskedImageConvolution = new Mat(), invMaskConvolution = new Mat();
        //this gives a weighted sum of data from points that are nearby a speck but outside it
        Imgproc.filter2D(invMaskedImage, invMaskedImageConvolution, CvType.CV_16U, payneKernel);
        //this gives a weighted sum of the nearby points
        Imgproc.filter2D(invMask, invMaskConvolution, CvType.CV_16U, payneKernel);
        //this gives us the "smoothed" values
        Mat smoothedData = new Mat();
        Core.divide(invMaskedImageConvolution, invMaskConvolution, smoothedData, 255.0);

        //remove smoothed data that lies outside of specks
        Mat smoothedDataMasked = new Mat();
        Core.bitwise_or(smoothedData, smoothedData, smoothedDataMasked, mask);

        //combine the points that were in a speck and that were smoothed with points that were not in a speck and left as is
        Mat outputData = new Mat();
        Core.add(smoothedDataMasked, invMaskedImage, outputData);
        return new Pair<>(rawMat, outputData);
    }

    public enum Visualization {
        SaturateCast,
        NormalizeMinMax,
        EqualizeHistogram;
    }

    public static void outputNirDataAsGrayscaleImage(Mat nirData, String fileName) {
        outputNirDataAsGrayscaleImage(nirData, fileName, Visualization.NormalizeMinMax);
    }

    public static void outputNirDataAsGrayscaleImage(Mat nirData, String fileName, Visualization visualization) {
        Mat nirData8bit = new Mat();
        switch (visualization) {
            case SaturateCast:
                //elements bigger than 255 will be rounded down to 255
                nirData.convertTo(nirData8bit, CvType.CV_8U);
                break;
            case NormalizeMinMax:
                //linearly scales all values between the min and max to fit range 0 to 255
                Core.normalize(nirData, nirData8bit, 0, 255, NORM_MINMAX, CvType.CV_8U);
                break;
            case EqualizeHistogram:
                //redistributes the range of data values such that they are spread roughly equally over the range 0 to 255
                nirData.convertTo(nirData8bit, CvType.CV_8U , 1.0/256.0);
                Imgproc.equalizeHist(nirData8bit, nirData8bit);
                break;
        }
        Imgcodecs.imwrite(fileName, nirData8bit);
    }
}
