package com.example.irina.test;

import java.nio.charset.StandardCharsets;

/**
 * Created by irina on 04.12.17.
 */

public class Stitching {
    private static byte[] stringToBytes(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }

    public native void stitchImages(byte[] data, int width, int height, byte[] data2, int width2, int height2,long matAddr);

    public void StitchImages(String imgPath, String imgPath2,long matAddr) {
        stitchImages(stringToBytes(imgPath), 0, 0, stringToBytes(imgPath2), 0, 0,matAddr);
    }
}
