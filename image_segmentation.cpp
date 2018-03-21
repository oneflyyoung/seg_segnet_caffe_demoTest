#include "CommonDef.h"
#include "image_segmentation.h"
#include <list>
#include "cvCommon.h"

using namespace std;

#define PATCH_SIZE 500
#define PATCH_OVERLAP      100

//do multi-scale for image
static void get_multi_scale_images(const Mat &img, list<Mat> *lstImages)
{
    int height = 500;
    while (height < 3000)
    {
        int img_width = (float)height / img.rows * img.cols + 0.5;
        int img_height = height;
        Mat temp;
        resize(img, temp, Size(img_width, img_height));
        lstImages->push_back(temp);

        if (height > img.rows)
            break;

        height += PATCH_SIZE;        
    }
}

static void get_multi_scale_images_for_ovlap(const Mat &img, list<Mat> *lstImages)
{
    int height = 500;
    while (height < 3000)
    {
        int img_width = (float)height / img.rows * img.cols + 0.5;
        int img_height = height;
        Mat temp;
        resize(img, temp, Size(img_width, img_height));
        lstImages->push_back(temp);

        if (height > img.rows)
            break;

        height += 2* (PATCH_SIZE - 2 * PATCH_OVERLAP);
    }
}

static void merge_output_result(Mat &output, Mat &output_scale_img)
{
    for (int y=0; y<output.rows; y++)
    {
        uchar *pOutput = output.data + y * output.cols;
        uchar *pScale = output_scale_img.data + y * output.cols;
        for (int x=0; x<output.cols; x++)
        {
            if (pOutput[x] > pScale[x])
                pOutput[x] = pScale[x];
        }
    }
}

//segmentation
void image_segmentation::process_image(Classifier &classifier,  Mat &img, Mat &output)
{
    int img_width = img.cols;	//宽
    int img_height = img.rows;	//高

    //根据需要调整图像大小
    if (img_width < PATCH_SIZE || img_height < PATCH_SIZE)
    {
        if (img_width < img_height)
        {
            img_height = (float)PATCH_SIZE / img_width * img_height + 0.5;
            img_width = PATCH_SIZE;
        }
        else
        {
            img_width = (float)PATCH_SIZE / img_height * img_width + 0.5;
            img_height = PATCH_SIZE;
        }

        resize(img, img, Size(img_width, img_height));
        resize(output, output, Size(img_width, img_height));
    }

    img_width = img.cols;	//宽
    img_height = img.rows;	//高
    for (int y=0; y<img_height; y+=PATCH_SIZE)
    {
        for (int x=0; x<img_width; x+=PATCH_SIZE)
        {
            Rect rect(x, y, PATCH_SIZE, PATCH_SIZE);
            if (rect.x + PATCH_SIZE > img_width)
                rect.x = img_width - PATCH_SIZE;
            if (rect.y + PATCH_SIZE > img_height)
                rect.y = img_height - PATCH_SIZE;

            //copy image data
            Mat inputPatch, outputPatch;
            img(rect).copyTo(inputPatch);
            //predict
            classifier.Predict(inputPatch, outputPatch);

            //copy image data
            Mat dstroi = output(rect);
            outputPatch.convertTo(dstroi, dstroi.type());
        }
    }
}

void image_segmentation::process_image_with_overlap(Classifier &classifier,  Mat &img, Mat &output)
{
    int img_width = img.cols;	//宽
    int img_height = img.rows;	//高

    //根据需要调整图像大小
    if (img_width < PATCH_SIZE || img_height < PATCH_SIZE)
    {
        if (img_width < img_height)
        {
            img_height = (float)PATCH_SIZE / img_width * img_height + 0.5;
            img_width = PATCH_SIZE;
        }
        else
        {
            img_width = (float)PATCH_SIZE / img_height * img_width + 0.5;
            img_height = PATCH_SIZE;
        }

        resize(img, img, Size(img_width, img_height));
        resize(output, output, Size(img_width, img_height));
    }

    img_width = img.cols;	//宽
    img_height = img.rows;	//高
    for (int y=0; y<img_height; y+=PATCH_SIZE)
    {
        if (y != 0)
            y -= 2 * PATCH_OVERLAP;

        for (int x=0; x<img_width; x+=PATCH_SIZE)
        {
            if (x != 0)
                x -= 2 * PATCH_OVERLAP;

            Rect rect(x, y, PATCH_SIZE, PATCH_SIZE);
            if (rect.x + PATCH_SIZE > img_width)
                rect.x = img_width - PATCH_SIZE;
            if (rect.y + PATCH_SIZE > img_height)
                rect.y = img_height - PATCH_SIZE;

            //copy image data
            Mat inputPatch, outputPatch;
            img(rect).copyTo(inputPatch);

            //predict
            classifier.Predict(inputPatch, outputPatch);

            Rect srcRect(0, 0, outputPatch.cols, outputPatch.rows);

            //copy image data
            if (x != 0)
            {
                rect.x +=PATCH_OVERLAP;
                rect.width -= PATCH_OVERLAP;
                srcRect.x += PATCH_OVERLAP;
                srcRect.width -= PATCH_OVERLAP;
            }
            if (y != 0)
            {
                rect.y += PATCH_OVERLAP;
                rect.height -= PATCH_OVERLAP;
                srcRect.y += PATCH_OVERLAP;
                 srcRect.height -= PATCH_OVERLAP;
            }

#ifdef _DEBUG
            char temp_save_path[256] = {0};
            sprintf(temp_save_path, "../temp/%d_%d.png", y, x);
            imwrite(temp_save_path, outputPatch);
#endif

            Mat dstroi = output(rect);
            outputPatch(srcRect).convertTo(dstroi, dstroi.type());
        }
    }
}

void image_segmentation::process_image_with_low_resolution(Classifier &classifier, const Mat &img, Mat &output)
{
    const int low_resolution = 250;

    //resize the original image with a very low resolution
    int img_width = img.cols;	//宽
    int img_height = img.rows;	//高

    //根据需要调整图像大小
        if (img_width > img_height)
        {
            img_height = (float)low_resolution / img_width * img_height + 0.5;
            img_width = low_resolution;
        }
        else
        {
            img_width = (float)low_resolution / img_height * img_width + 0.5;
            img_height = low_resolution;
        }
        Mat img_resize;
        resize(img, img_resize, Size(img_width, img_height));

        //create temp input and temp output image
        Mat input_temp = cv::Mat(500, 500, img.type(), Scalar(0, 0, 0));
        Mat output_temp = cv::Mat(500, 500, CV_8UC1, Scalar(0));

        //copy image data to temp input image
        Rect dstRect(0, 0, img_width, img_height);
        Mat dstroi = input_temp(dstRect);
        img_resize.convertTo(dstroi, dstroi.type());

        //image segmentation
        process_image(classifier, input_temp, output_temp);

        Mat valid_output;       //the valid output patch
        output_temp(dstRect).copyTo(valid_output);

        //resize the output
        resize(valid_output, output, Size(img.cols, img.rows));
}

void image_segmentation::process_image_with_multiscale(Classifier &classifier, const Mat &img, Mat &output)
{
    //create output image
    output = cv::Mat(img.size(), CV_8UC1, Scalar(0));

#ifdef _USE_LOW_RESOLUTION
    process_image_with_low_resolution(classifier, img, output);
#endif

    //get multi-scale images
    list<Mat> lstImages;
    //get_multi_scale_images(img, &lstImages);
    get_multi_scale_images_for_ovlap(img, &lstImages);
#ifdef _DEBUG
        printf ("The number of multi-scale image is %d\n",  lstImages.size());
#endif

    //process multi-scale images
    int index = 0;
    list<Mat>::iterator iter = lstImages.begin();
    for (; iter != lstImages.end(); iter++)
    {
        cvCommon::AutoTone(*iter);
        Mat output_scale_img = cv::Mat(iter->size(), CV_8UC1, Scalar(0));
        //process_image(classifier, *iter, output_scale_img);
        process_image_with_overlap(classifier, *iter, output_scale_img);

#ifdef _DEBUG
        char temp_save_path[256] = {0};
        sprintf(temp_save_path, "../temp/scale_%d.png", index);
        imwrite(temp_save_path, output_scale_img);
        index++;
#endif

        //change size to the original image
        resize(output_scale_img, output_scale_img, output.size());
        if (iter == lstImages.begin())
        {
#ifdef _USE_LOW_RESOLUTION
            merge_output_result(output, output_scale_img);
#else
            output = output_scale_img;
#endif
        }
        else
         {
            //merge result
            merge_output_result(output, output_scale_img);
        }
    }
}









