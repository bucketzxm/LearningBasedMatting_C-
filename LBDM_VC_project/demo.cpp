#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

#include "lbdm.h"

using namespace std;
using namespace cv;


bool showPic(1), savePic(0);

////

bool showSavePic(std::string name, cv::Mat mat, bool flag_showPic, bool flag_savePic){
	//
	if(flag_showPic){
		cv::namedWindow(name);
		cv::imshow(name, mat);
	}
	if(flag_savePic){
		std::string filePath;
		filePath = std::string("tempPicture\\") + name + ".png";
		cv::imwrite(filePath, mat);
	}
	return true;
}

int main(){
	//
	Mat mat, trimap;
	mat = imread("inputPic\\plasticbag_150.png", IMREAD_COLOR);
	trimap = imread("inputTrimap\\plasticbag_150.png", IMREAD_GRAYSCALE);

	showSavePic("mat", mat, showPic, savePic);
	showSavePic("trimap", trimap, showPic, savePic);
	cv::waitKey(0);

	Mat alpha =  LBDM_Matting(mat, trimap);

	showSavePic("alpha", alpha, showPic, savePic);
	cv::waitKey(0);

}