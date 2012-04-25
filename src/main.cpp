#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

int main(int argc, char* argv[]) {

	// Déclarations
	CvHaarClassifierCascade *pCascadeFrontal = 0, *pCascadeProfile = 0;	// le detecteur de visage 
	CvMemStorage *pStorage = 0;		// buffer mémoire expensible
	CvSeq *pFaceRectSeq;			// liste des visages detectés
	int i;

	/*/ Capture Webcam
	CvCapture *capture;
	capture = cvCreateCameraCapture(CV_CAP_ANY);
	pInpImg = cvQueryFrame(capture);*/

	// Initialisations
	//IplImage* pInpImg = (argc > 1) ? cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR) : 0;
	IplImage *pInpImg = cvLoadImage("D:/FaceRecognition/resources/COD2.jpg", CV_LOAD_IMAGE_COLOR);
	pStorage = cvCreateMemStorage(0);

	pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/FaceRecognition/resources/haarcascade/haarcascade_frontalface_default.xml",0,0,0);
	//pCascadeFrontal = (CvHaarClassifierCascade *) cvLoad ("D:/FaceRecognition/resources/haarcascade/haarcascade_frontalface_alt_tree.xml",0,0,0);
	pCascadeProfile = (CvHaarClassifierCascade *) cvLoad ("D:/FaceRecognition/resources/haarcascade/haarcascade_profileface.xml",0,0,0);

	// On valide que tout a bien été initialisé correctement
	if (!pInpImg || !pStorage || !pCascadeFrontal || !pCascadeProfile) {
		printf("L'initilisation a echoue");
		exit(-1);
	}

	// Affiche une fenêtre pour l'affichage des visages
	cvNamedWindow("Fenetre de Haar", CV_WINDOW_NORMAL);
	cvShowImage("Fenetre de Haar", pInpImg);
	cvWaitKey(50);

	// Detection de visage DE FACE dans l'image
	pFaceRectSeq = cvHaarDetectObjects
		(pInpImg, pCascadeFrontal, pStorage,
		1.1,	// augmente l'échelle de recherche de 10% à chaque passe [1.0-1.4] : plus c'est grand, plus c'est rapide
		3,	// met de côté les groupes plus petit que 3 détections [0-4] : plus c'est petit, plus il y aura de "hits"
		/*0,*/ CV_HAAR_DO_CANNY_PRUNING,	// [0] : explore tout ; [1] : abandonne les régions non candidates à contenir un visage
		cvSize(0, 0));	// utilise les paramètres XML par défaut (24, 24) pour la plus petite echelle de recherche

	// Dessine un rectangle autour de chaque visage detecté
	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0);
		
		// Floutage 
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}
	cvShowImage("Fenetre de Haar", pInpImg);
	cvWaitKey(50);

	// Detection de visage DE PROFIL dans l'image
	pFaceRectSeq = cvHaarDetectObjects
		(pInpImg, pCascadeProfile, pStorage,
		1.4,	// augmente l'échelle de recherche de 10% à chaque passe [1.0-1.4] : plus c'est grand, plus c'est rapide
		3,	// met de côté les groupes plus petit que 3 détections [0-4] : plus c'est petit, plus il y aura de "hits"
		/*0,*/ CV_HAAR_DO_CANNY_PRUNING,	// abandonne les régions non candidates à contenir un visage
		cvSize(0, 0));	// utilise les paramètres XML par défaut (24, 24) pour la plus petite echelle de recherche

	// Dessine un rectangle autour de chaque visage detecté
	for (i=0 ; i < (pFaceRectSeq ? pFaceRectSeq->total : 0) ; i++) {
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(pInpImg, pt1, pt2, CV_RGB(255,165,0), 3, 4, 0);
		
		// Floutage 
		cvSetImageROI(pInpImg, *r);
		cvSmooth(pInpImg, pInpImg, CV_GAUSSIAN, 5, 3);
		cvResetImageROI(pInpImg);
	}

	// Affiche la détection de visage
	cvShowImage("Fenetre de Haar", pInpImg);
	cvWaitKey(0);
	cvDestroyWindow("Fenetre de Haar");

	// Libère les ressources
	//cvReleaseCapture(&capture); // Capture Webcam
	cvReleaseImage(&pInpImg);
	if (pCascadeFrontal) cvReleaseHaarClassifierCascade(&pCascadeFrontal);
	if (pCascadeProfile) cvReleaseHaarClassifierCascade(&pCascadeProfile);
	if (pStorage) cvReleaseMemStorage(&pStorage);
}