package com.example.imagepro;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class imageClassification {

    //Add tensorflow im
    // using this interpreter we will load model and predict on frame
    private Interpreter interpreter;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3;  //for rgb
    private int IMAGE_MEAN=0;
    private float IMAGE_STD=255.0f; //  IMAGE_STD is used to scale image from 0->255 to 0->1
    // used to initialize gpu in interpreter

    private GpuDelegate gpuDelegate;
    private int height=0;
    private int width=0;

    // Define 2 color red and green
    //red for malignant
    // green for benign
    //                              R   G B  alpha
    private Scalar red = new Scalar(255,0,0,50);
    private Scalar green = new Scalar(0,255,0,50);



    imageClassification(AssetManager assetManager, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        //used to set gpu or number of threads
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(6);   // set the number of threads according to your phone
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
    }
    public Mat recognizeImage(Mat mat_image){
        // input image in landscape mode
        // convert it into portrait mode
        //to convert it rotate image by 90 degree

        Mat rotate_mat_image = new Mat();
        Core.flip(mat_image.t(),rotate_mat_image,1);

        // define height and width of original bitmap
        height= rotate_mat_image.height();
        width=rotate_mat_image.width();

       // Now we draw a rectangle of size (400,400) in center
        // Before that crop that part of the image
        //                              starting point                 width       height
        Rect roi_cropped=new Rect((width-400)/2,(height-400)/2,400,400);
        Mat cropped_image = new Mat(rotate_mat_image,roi_cropped);
        // fro prediction use this image

        // Convert Mat image image to bitmap image
        Bitmap bitmap =null;
        bitmap=Bitmap.createBitmap(cropped_image.cols(),cropped_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(cropped_image,bitmap);

        //resize bitmap image to input size of model

        Bitmap scaledBitmap =Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

        // convert selectedBitmap to ByteBuffer
        ByteBuffer byteBuffer=createBitMapToByteBuffer(scaledBitmap);

        // create an input and output for interpreter
        float[][] output = new float[1][1];
        Object[] out = new Object[1];
        out[0] = output;

        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // now pass it througth the interpreter
        interpreter.run(byteBuffer,output);
        // we are not using out and input as its used for multiple input and output
        // example interpreter.runForMultipleInputsOutputs(input,out);

        // to see value of output
        Log.d("imageClassification","out"+ Arrays.deepToString(output));

        //close to 0 benign
        //close to 1 malignant
        // set threshold to get better result

        // extract value from output
        float val_prediction =(float)Array.get(Array.get(output,0),0);

        // set threshold
        if(val_prediction>.4){
            // put text that malignant is detected
            // should be on original rotated frame
            //           image mat                 text                         starting point                                    color
            Imgproc.putText(rotate_mat_image,"Malignant is detected",new Point((width-400)/2+30,80),3,1,red,2);
            // draw rectangle in red
           Imgproc.rectangle(rotate_mat_image,new Point((width-400)/2,(height-400)/2),new Point((width+400)/2,(height+400)/2),red,2);
        }else{
            // put text that Benign is detected
            // should be on original rotated frame
            //           image mat                 text                         starting point                                    color
            Imgproc.putText(rotate_mat_image,"Benign is detected",new Point((width-400)/2+30,80),3,1,green,2);
            // draw rectangle in red
            Imgproc.rectangle(rotate_mat_image,new Point((width-400)/2,(height-400)/2),new Point((width+400)/2,(height+400)/2),green,2);
        }

        //returned image should be in landscape mode
        // rotate image  -90by degree
        Core.flip(rotate_mat_image.t(),rotate_mat_image,0);
        return  rotate_mat_image;
    }

    private ByteBuffer createBitMapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4*INPUT_SIZE*INPUT_SIZE*PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE*INPUT_SIZE];

        scaledBitmap.getPixels(intValues,0,scaledBitmap.getWidth(),0,0,scaledBitmap.getWidth(),scaledBitmap.getHeight());

        int pixel=0;
        for (int i=0; i<INPUT_SIZE; i++){
            for (int j=0; j<INPUT_SIZE; j++){
                final int val=intValues[pixel++];

                // set value of byte buffer
                //image_mean and image_std is used to convert image pixel from 0->255 to 0->1 to
                byteBuffer.putFloat((((val>>16) & 0xFF)- IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val>>8) & 0xFF)- IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)- IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }


    // this is used to load model
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws  IOException {

        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
}
