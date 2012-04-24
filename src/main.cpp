#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

int main(int argc, char* argv[]) {
        // Open the file.
        IplImage *img = cvLoadImage("photo.jpg");
        if (!img) {
                printf("Error: Couldn't open the image file.\n");
                return 1;
        }

        // Display the image.
		cvNamedWindow("Image:", CV_WINDOW_FULLSCREEN);
        cvShowImage("Image:", img);

        // Wait for the user to press a key in the GUI window.
        cvWaitKey(0);

        // Free the resources.
        cvDestroyWindow("Image:");
        cvReleaseImage(&img);
        
        return 0;
}