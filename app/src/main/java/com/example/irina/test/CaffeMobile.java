package com.example.irina.test;

/**
 * Created by irina on 06.11.17.
 */

import java.nio.charset.StandardCharsets;

public class CaffeMobile {
    private static byte[] stringToBytes(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }

    public native void setNumThreads(int numThreads);

    public native int loadModel(String modelPath, String weightsPath);  // required

    public native void predictImage(byte[] data, int width, int height, String LUT_file,long matAddr);

    public void predictImage(String imgPath, String LUT_file,long matAddr) {
        predictImage(stringToBytes(imgPath), 0, 0, LUT_file,matAddr);
    }

}
