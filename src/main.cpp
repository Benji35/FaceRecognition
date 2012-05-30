#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

void detectFaces(IplImage *img, CvHaarClassifierCascade *cascade, CvMemStorage *storage, CvSeq *faceRectSeq, CvScalar color);

int main(int argc, char* argv[]) {
	// D�clarations
	CvHaarClassifierCascade *pCascadeFrontal = 0, *pCascadeProfile = 0;	// le detecteur de visage 
	CvMemStorage *pStorage = 0;		// buffer m�moire expensible
	CvSeq *pFaceRectSeq = 0;		// liste des visages detect�s
	IplImage *pInpImg;				// image � analyser
	char key = 'r';					// touche utilis�e

	// Capture Webcam
	CvCapture *capture;
	capture = cvCreateCameraCapture(CV_CAP_ANY);
	//pInpImg = cvQueryFrame(capture);

	// Initialisations
	//IplImage* pInpImg = (argc > 1) ? cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR) : 0;
	//IplImage *pInpImg = cvLoadImage("D:/FaceRecognition/resources/COD2.jpg", CV_LOAD_IMAGE_COLOR);
	pStorage = cvCreateMemStorage(0);

	//pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/FaceRecognition/resources/haarcascade/haarcascade_frontalface_default.xml",0,0,0);
	pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/FaceRecognition/resources/haarcascade/haarcascade_frontalface_alt.xml",0,0,0);
	pCascadeProfile = (CvHaarClassifierCascade *) cvLoad ("D:/FaceRecognition/resources/haarcascade/haarcascade_profileface.xml",0,0,0);

	// On valide que tout a bien �t� initialis� correctement
	if (!pStorage || !pCascadeFrontal || !pCascadeProfile) {
		printf("L'initilisation a echoue");
		exit(-1);
	}

	// D�tection en boucle des visages
	while (1) {
		pInpImg = cvQueryFrame(capture);

		// Affiche une fen�tre pour l'affichage des visages
		cvNamedWindow("Fenetre de Haar", CV_WINDOW_NORMAL);
		cvShowImage("Fenetre de Haar", pInpImg);
		key = cvWaitKey(50);

		// D�tection des visages de face
		detectFaces(pInpImg, pCascadeFrontal, pStorage, pFaceRectSeq, CV_RGB(0,255,0));
		cvShowImage("Fenetre de Haar", pInpImg);
		if (key != 'p')	key = cvWaitKey(500);
		else cvWaitKey(50);
	
		// D�tection des visages de profil
		detectFaces(pInpImg, pCascadeProfile, pStorage, pFaceRectSeq, CV_RGB(255,165,0));
		cvShowImage("Fenetre de Haar", pInpImg);
		if (key != 'p')	key = cvWaitKey(500);
		else cvWaitKey(1000);

		// Pause sur image avec la touche 'p' ; 'r' pour continuer
		if (key == 'p') {
			while (key != 'r') {
				key = cvWaitKey(20);
			}
		}
	}

	// Lib�re les ressources
	cvReleaseCapture(&capture); // Capture Webcam
	//cvReleaseImage(&pInpImg);
	if (pCascadeFrontal) cvReleaseHaarClassifierCascade(&pCascadeFrontal);
	if (pCascadeProfile) cvReleaseHaarClassifierCascade(&pCascadeProfile);
	if (pStorage) cvReleaseMemStorage(&pStorage);
}

void detectFaces(IplImage *img, CvHaarClassifierCascade *cascade, CvMemStorage *storage, CvSeq *faceRectSeq, CvScalar color) {
	// Detection de visage dans l'image
	faceRectSeq = cvHaarDetectObjects
		(img, cascade, storage,
		1.15,	// augmente l'�chelle de recherche de 10% � chaque passe [1.0-1.4] : plus c'est grand, plus c'est rapide
		4,	// met de c�t� les groupes plus petit que 3 d�tections [0-4] : plus c'est petit, plus il y aura de "hits"
		/*0,*/ CV_HAAR_DO_CANNY_PRUNING,	// [0] : explore tout ; [1] : abandonne les r�gions non candidates � contenir un visage
		cvSize(0, 0));	// utilise les param�tres XML par d�faut (24, 24) pour la plus petite echelle de recherche

	// Dessine un rectangle autour de chaque visage detect�
	for (int i=0 ; i < (faceRectSeq ? faceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(faceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		//cvRectangle(img, pt1, pt2, color, 2, 4, 0);
		
		// Floutage 
		cvSetImageROI(img, *r); // Cr�ation d'une r�gion d'int�r�t (ROI), les traitements appliqu�s sur l'image se feront que sur le rectangle sp�cifi� ici
		int kernel_size = 0;
		(r->width <= r->height) ? kernel_size = r->width : kernel_size = r->height;
		if (kernel_size%2 == 0) {
			kernel_size++;
		}		
		cvSmooth(img, img, CV_GAUSSIAN, kernel_size);
		cvResetImageROI(img);
	}
}