package com.example.irina.test;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.ClipData;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

import com.example.irina.test.Stitching;

import static org.opencv.android.Utils.matToBitmap;

public class StitchActivity extends Activity {
    private Button btnSelect1;
    private Button btnSelect2;
    private Bitmap bmp1;
    private Bitmap bmp2;
    private Bitmap yourbitmap;
    private LinearLayout lnrImages;
    private Uri fileUri;
    String imageEncoded;
    List<String> imagesPathList;
    private ImageView ivOne, ivTwo,ivResult;
    private static final String LOG_TAG = "StitchingActivity";
    private static final int REQUEST_IMAGE_SELECT1 = 200;
    private static final int REQUEST_IMAGE_SELECT2 = 300;

    private Stitching stitcher;
    private String imgPath1 = null;
    private String imgPath2 = null;

    private Mat resultMat;

    /*static{
        System.loadLibrary("mystitcher");
        System.loadLibrary("opencv_java3");
    }*/

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_stitch);

        ivOne = (ImageView) findViewById(R.id.ivOne);
        ivTwo = (ImageView) findViewById(R.id.ivTwo);
        ivResult = (ImageView) findViewById(R.id.ivResult);

        lnrImages = (LinearLayout)findViewById(R.id.lnrImages);

        btnSelect1 = (Button) findViewById(R.id.btnSelect1);
        btnSelect1.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, REQUEST_IMAGE_SELECT1);
            }
        });

        btnSelect2 = (Button) findViewById(R.id.btnSelect2);
        btnSelect2.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(i, REQUEST_IMAGE_SELECT2);
            }
        });

        stitcher = new Stitching();
        resultMat = new Mat();


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if ((requestCode == REQUEST_IMAGE_SELECT1 || requestCode == REQUEST_IMAGE_SELECT2) && resultCode == RESULT_OK) {

            if (requestCode == REQUEST_IMAGE_SELECT1) {
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                Cursor cursor = StitchActivity.this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgPath1 = cursor.getString(columnIndex);
                cursor.close();
                bmp1 = BitmapFactory.decodeFile(imgPath1);
                Log.d(LOG_TAG, imgPath1);
                Log.d(LOG_TAG, String.valueOf(bmp1.getHeight()));
                Log.d(LOG_TAG, String.valueOf(bmp1.getWidth()));
                ivOne.setImageBitmap(bmp1);
            } else {
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                Cursor cursor = StitchActivity.this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                imgPath2 = cursor.getString(columnIndex);
                cursor.close();
                bmp2 = BitmapFactory.decodeFile(imgPath2);
                Log.d(LOG_TAG, imgPath2);
                Log.d(LOG_TAG, String.valueOf(bmp2.getHeight()));
                Log.d(LOG_TAG, String.valueOf(bmp2.getWidth()));
                ivTwo.setImageBitmap(bmp2);
            }

            if(imgPath1!=null && imgPath2!=null){
                try {
                    stitcher.StitchImages(imgPath1, imgPath2, resultMat.getNativeObjAddr());

                    Bitmap bmpResult = Bitmap.createBitmap(resultMat.cols(), resultMat.rows(),Bitmap.Config.ARGB_8888);
                    matToBitmap(resultMat, bmpResult);

                    ivResult.setImageBitmap(bmpResult);
                }catch(Exception e){
                    Log.d(LOG_TAG, e.getMessage());}
            }

        } else {
            btnSelect1.setEnabled(true);
            btnSelect2.setEnabled(true);
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

}
