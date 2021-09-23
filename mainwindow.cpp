#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

#include <string>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <math.h>

#include "NormalizationLib.h"
#include "DispLib.h"
#include "histograms.h"
#include "RegionU16Lib.h"
#include "StringFcLib.h"

#include "segmentMK.h"

#include "mazdaroi.h"
#include "mazdaroiio.h"

#include <tiffio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef MazdaRoi<unsigned int, 2> MR2DType;

using namespace boost;
using namespace std;
using namespace boost::filesystem;
using namespace cv;

const double pi = 3.14159265359;

//------------------------------------------------------------------------------------------------------------------------------
//           Out of Calss functions
//------------------------------------------------------------------------------------------------------------------------------
int RenumberMask(cv::Mat MaskMaster, cv::Mat MaskSlave)
{
    if(MaskMaster.empty() || MaskSlave.empty())
      return  -1;
    if(MaskMaster.type() != CV_16U || MaskSlave.type() != CV_16U)
      return  -2;
    if(MaskMaster.cols != MaskSlave.cols || MaskMaster.rows != MaskSlave.rows)
      return  -3;

    int maxX = MaskMaster.cols;
    int maxY = MaskMaster.rows;
    int maxXY = maxX * maxY;


    uint16_t *wMaskMaster = (uint16_t *)MaskMaster.data;
    uint16_t *wMaskSlave = (uint16_t *)MaskSlave.data;

    for(int i = 0; i < maxXY; i++)
    {
        if(*wMaskSlave)
            *wMaskSlave = *wMaskMaster;
        wMaskMaster++;
        wMaskSlave++;
    }
    return 1;
}
//------------------------------------------------------------------------------------------------------------------------------
bool SaveQMaZdaROI(cv::Mat Mask, std::string FileName)
{
    if(Mask.empty())
        return 0;
    if(Mask.type() != CV_16U)
        return 0;
    int maxX = Mask.cols;
    int maxY = Mask.rows;
    int maxXY = maxX * maxY;

    int *RoiSizes = new int[65536];
    for(unsigned int k = 0; k<65536; k++)
    {
        RoiSizes[k] = 0;
    }
    uint16_t *wMask = (uint16_t *)Mask.data;
    for(int i = 0; i< maxXY; i++)
    {
        RoiSizes[*wMask]++;
        wMask++;
    }

    vector <MR2DType*> ROIVect;
    int begin[MR2DType::Dimensions];
    int end[MR2DType::Dimensions];
    begin[0] = 0;
    begin[1] = 0;
    end[0] = maxX-1;
    end[1] = maxY-1;

    MR2DType *ROI;


    for(int roiNr = 1; roiNr <= 65535; roiNr++)
    {
        if(RoiSizes[roiNr])
        {
            ROI = new MR2DType(begin, end);

            MazdaRoiIterator<MR2DType> iteratorKL(ROI);
            wMask = (uint16_t *)Mask.data;
            while(! iteratorKL.IsBehind())
            {
                if (*wMask == roiNr)
                    iteratorKL.SetPixel();
                ++iteratorKL;
                wMask++;
            }

            ROI->SetName("R" + ItoStrLZ(roiNr,5));
            ROI->SetColor(RegColorsRGB[(roiNr-1)%16]);

            ROIVect.push_back(ROI);
        }
    }

    MazdaRoiIO<MR2DType>::Write(FileName, &ROIVect, NULL);
    while(ROIVect.size() > 0)
    {
         delete ROIVect.back();
         ROIVect.pop_back();
    }
    delete [] RoiSizes;
    return 1;
}

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          constructor Destructor
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->comboBoxShowMode->addItem("Mask");
    ui->comboBoxShowMode->addItem("Contour");
    ui->comboBoxShowMode->setCurrentIndex(0);

    ui->comboBoxBckRemovalType->addItem("none");

    ui->comboBoxBckRemovalType->addItem("MaSt");

    ui->comboBoxBckRemovalType->addItem("MaKo2");

    ui->comboBoxBckRemovalType->addItem("Atoselect");
    ui->comboBoxBckRemovalType->setCurrentIndex(3);


    allowMoveTile = 1;
}

MainWindow::~MainWindow()
{
    delete ui;
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          CLASS FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::OpenImageFolder()
{
    path ImageFolder( ui->lineEditImageFolder->text().toStdWString());

    ui->listWidgetImageFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(ImageFolder))
    {
        regex FilePattern(ui->lineEditRegexImageFile->text().toStdString());
        if (!regex_match(FileToProcess.path().filename().string().c_str(), FilePattern ))
            continue;
        path PathLocal = FileToProcess.path();
        if (!exists(PathLocal))
        {
            ui->textEditOut->append(QString::fromStdString(PathLocal.filename().string() + " File not exists" ));
            break;
        }
        ui->listWidgetImageFiles->addItem(PathLocal.filename().string().c_str());
    }

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ReadImage()
{
    if(ui->checkBoxAutocleanOut->checkState())
        ui->textEditOut->clear();
    int flags;
    //if(ui->checkBoxLoadAnydepth->checkState())
    //    flags = CV_LOAD_IMAGE_ANYDEPTH;
    //else
    //    flags = IMREAD_COLOR;
    flags = IMREAD_COLOR;
    ImIn = imread(FileToOpen.string(), flags);
    if(ImIn.empty())
    {
        ui->textEditOut->append("improper file");
        return;
    }
    ReffMask = Mat::zeros(ImIn.rows,ImIn.cols,CV_16U);
    LesionMask = Mat::zeros(ImIn.rows,ImIn.cols,CV_16U);
    string extension = FileToOpen.extension().string();

    if((extension == ".tif" || extension == ".tiff") && ui->checkBoxShowTiffInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(TiffFilePropetiesAsText(FileToOpen.string())));

    if(ui->checkBoxShowMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImIn)));
    LoadReffMask();
    if(ui->checkBox3Masks->checkState())
        MaskFusion();
    else
        LoadLesionMask();
    ScaleImMiniature();
    AdiustTilePositionSpinboxes();
    ProcessImages();
    GetTile();
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::MaskFusion()
{
    LesionMask1 =  LoadLesionMask("MaKo.bmp");
    LesionMask2 =  LoadLesionMask("_seg_MaSt.tiff");
    LesionMask3 =  LoadLesionMask("MiKo.bmp");

    Mat CombinedMask = Combine3RegionsTo8Bit(LesionMask1, LesionMask2, LesionMask3);
    if(CombinedMask.empty())
        return;
    int combinedMaskCount = connectedComponents(CombinedMask,LesionMask,8,CV_16U);

    RenumberMask(LesionMask, LesionMask1);
    RenumberMask(LesionMask, LesionMask2);
    RenumberMask(LesionMask, LesionMask3);

    MultiRegionsParams LesionMask1Params(LesionMask1);
    MultiRegionsParams LesionMask2Params(LesionMask2);
    MultiRegionsParams LesionMask3Params(LesionMask3);

    for(unsigned short k = 1; k <= combinedMaskCount; k++)
    {
        RegionParams Lesion1RegParams = LesionMask1Params.GetRegionParams(k);
        RegionParams Lesion2RegParams = LesionMask2Params.GetRegionParams(k);
        RegionParams Lesion3RegParams = LesionMask3Params.GetRegionParams(k);

        if(!((Lesion1RegParams.area && Lesion2RegParams.area) ||
            (Lesion1RegParams.area && Lesion3RegParams.area) ||
            (Lesion2RegParams.area && Lesion3RegParams.area)))
            DeleteRegionFromImage(LesionMask,  k);
    }
}

//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::LoadReffMask()
{
    QString ImageFolderQStr = ui->lineEditImageFolder->text();
    path maskFilePath = ImageFolderQStr.toStdWString();
    maskFilePath.append(FileToOpen.stem().string() + ".png");

    if(!exists(maskFilePath))
    {
        ui->textEditOut->append("image file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return;
    }


    if(exists(maskFilePath))
    {
        ReffMask = imread(maskFilePath.string(), cv::IMREAD_ANYDEPTH);
        if(ReffMask.type() != CV_16U)
        {
            ui->textEditOut->append("mask format improper");
            ReffMask.convertTo(ReffMask, CV_16U);
        }

        if(ReffMask.empty())
        {
            ReffMask = Mat::zeros(ImIn.rows, ImIn.cols, CV_16U);
            ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + "cannot be read");
            ui->textEditOut->append("empty mask was created");
        }
    }
    else
    {
        ReffMask = Mat::zeros(ImIn.rows, ImIn.cols, CV_16U);
        ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        ui->textEditOut->append("empty mask was created");
    }
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::LoadLesionMask()
{
    QString ImageFolderQStr = ui->lineEditImageFolder->text();
    path maskFilePath = ImageFolderQStr.toStdWString();
    maskFilePath.append(FileToOpen.stem().string() + ui->lineEditPostFix->text().toStdString());

    if(!exists(maskFilePath))
    {
        ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return;
    }

    Mat LesionMaskTemp;
    LesionMask = Mat::zeros(ImIn.rows, ImIn.cols, CV_16U);
    if(exists(maskFilePath))
    {
        LesionMaskTemp = imread(maskFilePath.string(), cv::IMREAD_ANYDEPTH);
        if(LesionMaskTemp.type() != CV_8U)
        {
            ui->textEditOut->append("lesion mask format improper");

            ui->textEditOut->append("empty mask was created");
            return;
        }

        if(LesionMaskTemp.empty())
        {

            ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + "cannot be read");
            ui->textEditOut->append("empty mask was created");
            return;
        }
        if(LesionMaskTemp.cols != ImIn.cols || LesionMaskTemp.rows != ImIn.rows)
        {

            ui->textEditOut->append("improper lesion mask size");
            ui->textEditOut->append("empty mask was created");
            return;
        }
    }
    else
    {

        ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        ui->textEditOut->append("empty mask was created");
        return;
    }

    unsigned char *wLesionMaskTemp = (unsigned char *)LesionMaskTemp.data;
    unsigned short *wLesionMask = (unsigned short *)LesionMask.data;

    int maxXY = ImIn.cols * ImIn.rows;
    for(int i=0; i < maxXY; i++)
    {
        if(*wLesionMaskTemp > 0)
            *wLesionMask = 1;
        wLesionMask++;
        wLesionMaskTemp++;
    }

}
//------------------------------------------------------------------------------------------------------------------------------
cv::Mat MainWindow::LoadLesionMask(std::string PostFix)
{

    Mat LesionMaskOut;
    QString ImageFolderQStr = ui->lineEditImageFolder->text();
    path maskFilePath = ImageFolderQStr.toStdWString();
    maskFilePath.append(FileToOpen.stem().string() + PostFix );//+ ".bmp");

    if(!exists(maskFilePath))
    {
        ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return LesionMaskOut;
    }


    Mat LesionMaskTemp;
    if(exists(maskFilePath))
    {
        LesionMaskTemp = imread(maskFilePath.string(), cv::IMREAD_ANYDEPTH);
        if(LesionMaskTemp.type() != CV_8U)
        {
            ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring())+" format improper");
            return LesionMaskOut;
        }

        if(LesionMaskTemp.empty())
        {
            ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + "cannot be read");
            return LesionMaskOut;
        }
        if(LesionMaskTemp.cols != ImIn.cols || LesionMaskTemp.rows != ImIn.rows)
        {
            ui->textEditOut->append("improper lesion mask size");
            return LesionMaskOut;
        }
    }
    else
    {

        ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return LesionMaskOut;
    }

    LesionMaskOut = Mat::zeros(ImIn.rows, ImIn.cols, CV_16U);
    unsigned char *wLesionMaskTemp = (unsigned char *)LesionMaskTemp.data;
    unsigned short *wLesionMaskOut = (unsigned short *)LesionMaskOut.data;

    int maxXY = ImIn.cols * ImIn.rows;
    for(int i=0; i < maxXY; i++)
    {
        if(*wLesionMaskTemp > 0)
            *wLesionMaskOut = 1;
        wLesionMaskOut++;
        wLesionMaskTemp++;
    }
    return LesionMaskOut;

}
//------------------------------------------------------------------------------------------------------------------------------
cv::Mat MainWindow::LoadLesionMask8bit(std::string PostFix)
{

    Mat LesionMaskOut;
    QString ImageFolderQStr = ui->lineEditImageFolder->text();
    path maskFilePath = ImageFolderQStr.toStdWString();
    maskFilePath.append(FileToOpen.stem().string() + PostFix+ ".bmp");

    if(!exists(maskFilePath))
    {
        ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return LesionMaskOut;
    }


    Mat LesionMaskTemp;
    if(exists(maskFilePath))
    {
        LesionMaskTemp = imread(maskFilePath.string(), cv::IMREAD_ANYDEPTH);
        if(LesionMaskTemp.type() != CV_8U)
        {
            ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring())+" format improper");
            return LesionMaskOut;
        }

        if(LesionMaskTemp.empty())
        {
            ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + "cannot be read");
            return LesionMaskOut;
        }
        if(LesionMaskTemp.cols != ImIn.cols || LesionMaskTemp.rows != ImIn.rows)
        {
            ui->textEditOut->append("improper lesion mask size");
            return LesionMaskOut;
        }
    }
    else
    {

        ui->textEditOut->append("mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return LesionMaskOut;
    }
    return LesionMaskTemp;
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowsScaledImage(Mat Im, string ImWindowName, double displayScale)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }

    Mat ImToShow;

    ImToShow = Im.clone();

    if (displayScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);
    imshow(ImWindowName, ImToShow);

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowImages()
{
    if(ImIn.empty())
        return;

    if(ReffMask.empty())
        return;

    int lineThickness = (int)ui->doubleSpinBoxImageScale->value();
    double scale = 1 / ui->doubleSpinBoxImageScale->value();

    int tileSizeX, tileSizeY;

    tileSizeX = ui->spinBoxTileSize->value();
    tileSizeY = ui->spinBoxTileSize->value();

    int tilePositionX = ui->spinBoxTileX->value();// * tileStep;
    int tilePositionY = ui->spinBoxTileY->value();// * tileStep;

    if(ui->checkBoxShowInput->checkState())
    {
        if(ui->checkBoxShowTileOnImage->checkState())
        {
            Mat ImToShow;
            ImIn.copyTo(ImToShow);
            rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);
            ShowsScaledImage(ImToShow, "Input Image", scale);
        }
        else
            ShowsScaledImage(ImIn, "Input Image", scale);
    }

    if(ui->checkBoxShowReffMask->checkState())
    {
        Mat ImToShow;
        ShowRegion(ReffMask).copyTo(ImToShow);
        rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);
        ShowsScaledImage(ImToShow, "Reff Mask", scale);
    }
    else
    {
        destroyWindow("Reff Mask");
    }


    if(ui->checkBoxShowLesionMask->checkState())
    {
        Mat ImToShow;
        ShowRegion(LesionMask).copyTo(ImToShow);
        rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);
        ShowsScaledImage(ImToShow, "Lesion mask", scale);

    }
    else
    {
        destroyWindow("Lesion mask");
    }

    if(ui->checkBoxShowLesionOnImage->checkState())
    {
        Mat ImToShow;
        if(ui->checkBoxShowMaskAsContour->checkState())
            ShowSolidRegionOnImage(GetContour5(Mask), ImIn).copyTo(ImToShow);
        else
            ShowSolidRegionOnImage(Mask, ImIn).copyTo(ImToShow);

        rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);

        int lesionIndex;
        if(ui->spinBoxLesionNr->value()>0 && NoCommonLiesionRegions.size() >= ui->spinBoxLesionNr->value())
            lesionIndex = NoCommonLiesionRegions[ui->spinBoxLesionNr->value()-1];
        else
            lesionIndex = 0;

        if(lesionIndex > 0)
        {
            RegionParams LesionRegParams = LesionRegionsParams.GetRegionParams(lesionIndex);

            rectangle(ImToShow, Rect(LesionRegParams.minX-3,
                                     LesionRegParams.minY-3,
                                     LesionRegParams.sizeX+6,
                                     LesionRegParams.sizeY+6),
                      Scalar(0.0, 0.0, 255.0, 0.0),
                      -1);
        }
//        lesionIndex = ReffLiesionRegions[ui->spinBoxReffNr->value()-1];

        if(ui->spinBoxReffNr->value() > 0 && ReffLiesionRegions.size() >= ui->spinBoxReffNr->value())
            lesionIndex = ReffLiesionRegions[ui->spinBoxReffNr->value()-1];
        else
            lesionIndex = 0;

        if(lesionIndex > 0)
        {
            RegionParams ReffRegParams = RefRegionsParams.GetRegionParams(lesionIndex);

            rectangle(ImToShow, Rect(ReffRegParams.minX-10,
                                     ReffRegParams.minY-10,
                                     ReffRegParams.sizeX+20,
                                     ReffRegParams.sizeY+20),
                      Scalar(255.0, 0.0, 255.0, 0.0),
                      -1);
        }
        ShowsScaledImage(ImToShow,"Lesion mask on Image", scale);
    }
    else
    {
        destroyWindow("Lesion mask on Image");
    }

    if(ui->checkBoxShowLesion->checkState())
    {
        //Mat LesionImToShow;


    }

    if(ui->checkBoxShowBcg->checkState())
    {
        Mat ImToShow;
        ShowSolidRegionOnImage(MaskBcg, ImIn).copyTo(ImToShow);
        ShowsScaledImage(ImToShow,"Background on Image", scale);
    }
    else
    {
        destroyWindow("Background on Image");
    }

    if(ui->checkBoxShowCommon->checkState())
    {
        Mat ImToShow;
        ShowRegion(Mask).copyTo(ImToShow);
        rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);
        ShowsScaledImage(ImToShow, "Common mask", scale);
    }
    else
    {
        destroyWindow("Common mask");
    }



    Mat ImToShow;
    ShowSolidRegionOnImage(Mask, ImIn).copyTo(ImToShow);;
    rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0),
              10);


    ui->widgetImageWhole->paintBitmap(ImToShow);
    ui->widgetImageWhole->repaint();

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ProcessImages()
{
    string FileName = FileToOpen.stem().string();
    MaskBcg.release();


    switch(ui->comboBoxBckRemovalType->currentIndex())
    {
    case 1:
        MaskBcg = MaskBackgroundMaSt(ImIn);
        break;
    case 2:
        MaskBcg = MaskBackgroundMaKo2(ImIn);
        break;
    case 3:
        if(FileName.at(0) == '_')
            MaskBcg = MaskBackgroundMaSt(ImIn);
        else
            MaskBcg = MaskBackgroundMaKo2(ImIn);
        break;
    default:
        MaskBcg = Mat::ones(ImIn.rows,ImIn.cols, CV_16U);
        break;

    }
    if(ui->comboBoxBckRemovalType->currentIndex())
    {
        MaskPostProcess(MaskBcg,30);
    }

    Mat CombinedMaskTemp = Combine2RegionsTo8Bit(ReffMask, LesionMask);

    if(CombinedMaskTemp.empty())
        return;

    //int combinedMaskCount = DivideSeparateRegions(CombinedMask, 10);

    int combinedMaskCount = connectedComponents(CombinedMaskTemp,CombinedMask,8,CV_16U);

    if(ui->comboBoxBckRemovalType->currentIndex())
    {
        RemoveOverlapingRegions(CombinedMask, MaskBcg);

    }


    RenumberMask(CombinedMask, ReffMask);
    RenumberMask(CombinedMask, LesionMask);

    int maxX = CombinedMask.cols;
    int maxY = CombinedMask.rows;
    int maxXY = maxX * maxY;

    Mask =       Mat::zeros(maxY,maxX, CV_16U);
    CommonMask = Mat::zeros(maxY,maxX, CV_16U);

    unsigned short * wReffMask;
    unsigned short * wLesionMask;
    unsigned short * wMask;
    unsigned short * wCommonMask;

    wReffMask = (unsigned short *)ReffMask.data;
    wLesionMask = (unsigned short *)LesionMask.data;
    wMask = (unsigned short *)Mask.data;
    wCommonMask = (unsigned short *)CommonMask.data;

    for(int i = 0; i < maxXY; i++)
    {
        if(*wLesionMask && *wReffMask)
        {
            *wCommonMask = 1;
            *wMask = 4;
        }
        else
        {
            if(*wLesionMask)
            {
                *wMask = 2;
            }
            if(*wReffMask)
            {
                *wMask = 3;
            }
        }

        wReffMask++;
        wLesionMask++;
        wCommonMask++;
        wMask++;
    }
    RenumberMask(CombinedMask, CommonMask);

    RefRegionsParams.GetFromMat(ReffMask);
    LesionRegionsParams.GetFromMat(LesionMask);
    CommonRegionsParams.GetFromMat(CommonMask);


    MultiRegionsParams CombinedRegionParams(CombinedMask);

    reffRegionCount = RefRegionsParams.GetCountOfNonZeroArea();
    lesionRegionCount = LesionRegionsParams.GetCountOfNonZeroArea();
    commonRegionCount = CommonRegionsParams.GetCountOfNonZeroArea();
    int combinedRegionCount = CombinedRegionParams.GetCountOfNonZeroArea();
    ui->textEditOut->append("Combined Maskn count = " + QString::number(combinedMaskCount));
    ui->textEditOut->append("Combined region count = " + QString::number(combinedRegionCount));
    ui->textEditOut->append("Refference region count = " + QString::number(reffRegionCount));
    ui->textEditOut->append("Lession region count = " + QString::number(lesionRegionCount));
    ui->textEditOut->append("Common region count = " + QString::number(commonRegionCount));

    NoCommonLiesionRegions.clear();
    ReffLiesionRegions.clear();
    for(int k = 1; k <= combinedMaskCount; k++)
    {
        RegionParams LesionRegParam = LesionRegionsParams.GetRegionParams(k);
        RegionParams CommonRegParam = CommonRegionsParams.GetRegionParams(k);
        RegionParams ReffRegParam = RefRegionsParams.GetRegionParams(k);
        if(LesionRegParam.area && (CommonRegParam.area == 0))
            NoCommonLiesionRegions.push_back(k);
        if(ReffRegParam.area && (CommonRegParam.area == 0))
            ReffLiesionRegions.push_back(k);
    }

    if(NoCommonLiesionRegions.size())
    {
        ui->spinBoxLesionNr->setMinimum(1);
        ui->spinBoxLesionNr->setMaximum(NoCommonLiesionRegions.size());
        ui->checkBoxLesionValid->setEnabled(true);
        ui->spinBoxLesionNr->setEnabled(true);
    }
    else
    {
        ui->spinBoxLesionNr->setEnabled(false);
        ui->spinBoxLesionNr->setMinimum(0);
        ui->spinBoxLesionNr->setMaximum(0);

        ui->checkBoxLesionValid->setEnabled(false);
    }

    if(ReffLiesionRegions.size())
    {
        ui->spinBoxReffNr->setMinimum(1);
        ui->spinBoxReffNr->setMaximum(ReffLiesionRegions.size());
        ui->spinBoxReffNr->setEnabled(true);
    }
    else
    {
        ui->spinBoxReffNr->setEnabled(false);
        ui->spinBoxReffNr->setMinimum(0);
        ui->spinBoxReffNr->setMaximum(0);

        ui->checkBoxLesionValid->setEnabled(false);
    }


    aClassRefCount = 0;
    aClassTPCount = 0;
    aClassFNCount = 0;

    bClassRefCount = 0;
    bClassTPCount = 0;
    bClassFNCount = 0;

    cClassRefCount = 0;
    cClassTPCount = 0;
    cClassFNCount = 0;

    totalRefCount = 0;
    totalTPCount = 0;
    totalFNCount = 0;
    totalFPCount = 0;
    for(int k = 1; k <= combinedMaskCount; k++)
    {
        RegionParams LesionRegParam = LesionRegionsParams.GetRegionParams(k);
        RegionParams CommonRegParam = CommonRegionsParams.GetRegionParams(k);
        RegionParams ReffRegParam = RefRegionsParams.GetRegionParams(k);

        double refArea = (double)ReffRegParam.area / 38.44;
        double equRefDiameter = sqrt(refArea * 4 / pi);

        double lesionArea = (double)LesionRegParam.area / 38.44;
        double equLesionDiameter = sqrt(lesionArea * 4 / pi);

        if((equRefDiameter >= 1.0) && (equRefDiameter < 2.0))
        {
            aClassRefCount++;
            if(LesionRegParam.area)
            {
                aClassTPCount++;
            }
            else
            {
                aClassFNCount++;
            }
        }

        if((equRefDiameter >= 2.0) && (equRefDiameter < 5.0))
        {
            bClassRefCount++;
            if(LesionRegParam.area)
            {
                bClassTPCount++;
            }
            else
            {
                bClassFNCount++;
            }
        }

        if(equRefDiameter >= 5.0)
        {
            cClassRefCount++;
            if(LesionRegParam.area)
            {
                cClassTPCount++;
            }
            else
            {
                cClassFNCount++;
            }
        }

        if(equRefDiameter > 1.0)
        {
            totalRefCount++;
            if(LesionRegParam.area)
            {
                totalTPCount++;
            }
            else
            {
                totalFNCount++;
            }
        }

        if((CommonRegParam.area == 0) && (LesionRegParam.area != 0) && (equLesionDiameter > 0.8) && (equLesionDiameter < 10.0))
            totalFPCount++;



        if(CommonRegParam.area != 0)
        {

            /*
            double reffCirDia = sqrt(pow((double)(ReffRegParam.maxX - ReffRegParam.minX), 2.0) +
                                     pow((double)(ReffRegParam.maxY - ReffRegParam.minY), 2.0));

            double lesionCirDia = sqrt(pow((double)(LesionRegParam.maxX - LesionRegParam.minX), 2.0) +
                                       pow((double)(LesionRegParam.maxY - LesionRegParam.minY), 2.0));

            Mat LesionMaskSmall, ReffMaskSmall;

            ReffMask(Rect(ReffRegParam.minX, ReffRegParam.minY,
                          ReffRegParam.sizeX, ReffRegParam.sizeY)).copyTo(ReffMaskSmall);
            KeepOneRegion(ReffMaskSmall, k);

            RotatedRect fittedRectReff;

            fittedRectReff.angle = 0.0;
            fittedRectReff.center = Point2f((float)(ReffMaskSmall.cols)/2,(float)(ReffMaskSmall.rows)/2);
            fittedRectReff.size = Size2f(100.0,100.0);



            Mat ImTemp;
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            Mat pointsF;

            ReffMaskSmall.convertTo(ImTemp,CV_8U);
            findContours(ImTemp,contours,hierarchy,CV_RETR_LIST,CHAIN_APPROX_NONE);
            if(contours.size())
            {
                int maxSize = 0;
                int maxContour = 0;
                for(int i = 0;i<contours.size();i++)
                {
                    if(maxSize < contours[i].size())
                    {
                        maxSize = contours[i].size();
                        maxContour = i;
                    }
                }
                if(maxSize > 10)
                {
                    Mat(contours[maxContour]).convertTo(pointsF, CV_32F);
                    fittedRectReff = fitEllipse(pointsF);
                }
            }

            contours.clear();
            hierarchy.clear();
            ImTemp.release();

                // rotate images
            double reffEllAngle = fittedRectReff.angle;
            double reffEllAxLong = fittedRectReff.size.height;
            double reffEllAxShort = fittedRectReff.size.width;



            LesionMask(Rect(LesionRegParam.minX, LesionRegParam.minY,
                            LesionRegParam.sizeX, LesionRegParam.sizeX)).copyTo(LesionMaskSmall);
            KeepOneRegion(LesionMaskSmall, k);

            RotatedRect fittedRectLesion;

            fittedRectLesion.angle = 0.0;
            fittedRectLesion.center = Point2f((float)(LesionMaskSmall.cols)/2,(float)(LesionMaskSmall.rows)/2);
            fittedRectLesion.size = Size2f(100.0,100.0);

            LesionMaskSmall.convertTo(ImTemp,CV_8U);
            findContours(ImTemp,contours,hierarchy,CV_RETR_LIST,CHAIN_APPROX_NONE);
            if(contours.size())
            {
                int maxSize = 0;
                int maxContour = 0;
                for(int i = 0;i<contours.size();i++)
                {
                    if(maxSize < contours[i].size())
                    {
                        maxSize = contours[i].size();
                        maxContour = i;
                    }
                }
                if(maxSize > 10)
                {
                    Mat(contours[maxContour]).convertTo(pointsF, CV_32F);
                    fittedRectLesion = fitEllipse(pointsF);
                }
            }

            contours.clear();
            hierarchy.clear();
            ImTemp.release();

                // rotate images
            double lesionEllAngle = fittedRectLesion.angle;
            double lesionEllAxLong = fittedRectLesion.size.height;
            double lesionEllAxShort = fittedRectLesion.size.width;



            ui->textEditOut->append(QString::number(k) + "\t" +
                                    QString::number(reffCirDia) + "\t" +
                                    QString::number(reffEllAxLong) + "\t" +
                                    QString::number(reffEllAxShort) + "\t" +
                                    QString::number(reffEllAngle) + "\t" +
                                    QString::number(lesionCirDia) + "\t" +
                                    QString::number(lesionEllAxLong) + "\t" +
                                    QString::number(lesionEllAxShort) + "\t" +
                                    QString::number(lesionEllAngle) + "\t");
          */

        }
    }
    if(ui->checkBox3Masks->checkState())
    {
        RenumberMask(CombinedMask, LesionMask1);
        RenumberMask(CombinedMask, LesionMask2);
        RenumberMask(CombinedMask, LesionMask3);
    }
    ui->textEditOut->append(QString::fromStdWString(FileToOpen.stem().wstring()) + "\t" +
                            QString::number(aClassRefCount) + "\t" +
                            QString::number(aClassTPCount) + "\t" +
                            QString::number(aClassFNCount) + "\t" +
                            QString::number(bClassRefCount) + "\t" +
                            QString::number(bClassTPCount) + "\t" +
                            QString::number(bClassFNCount) + "\t" +
                            QString::number(cClassRefCount) + "\t" +
                            QString::number(cClassTPCount) + "\t" +
                            QString::number(cClassFNCount) + "\t" +
                            QString::number(totalRefCount) + "\t" +
                            QString::number(totalTPCount) + "\t" +
                            QString::number(totalFNCount) + "\t" +
                            QString::number(totalFPCount) + "\t");


    ShowImages();
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::GetTile()
{
    if(ReffMask.empty())
        return;
    if(LesionMask.empty())
        return;
    if(ImIn.empty())
        return;
    //int tileStep = ui->spinBoxTileStep->value();
    int tileSizeX, tileSizeY;
    tileSizeX = ui->spinBoxTileSize->value();
    tileSizeY = ui->spinBoxTileSize->value();


    int tilePositionX = ui->spinBoxTileX->value();// * tileStep;
    int tilePositionY = ui->spinBoxTileY->value();// * tileStep;

    //prevTilePositionX = tilePositionX;
    //prevTilePositionY = tilePositionY;

    //Mat TileIm, TileReffMask, TileLesionMask, TileMask;

    ImIn(Rect(tilePositionX, tilePositionY, tileSizeX, tileSizeY)).copyTo(TileIm);
    ReffMask(Rect(tilePositionX, tilePositionY, tileSizeX, tileSizeY)).copyTo(TileReffMask);
    LesionMask(Rect(tilePositionX, tilePositionY, tileSizeX, tileSizeY)).copyTo(TileLesionMask);
    Mask(Rect(tilePositionX, tilePositionY, tileSizeX, tileSizeY)).copyTo(TileMask);

    Mat ImShow;
    if(TileLesionMask.empty() || TileIm.empty())
        return;
    //TileIm.copyTo(ImShow);

    switch(ui->comboBoxShowMode->currentIndex())
    {
    case 1:
        ImShow = ShowTransparentRegionOnImage(GetContour5(TileMask), TileIm, ui->spinBoxTransparency->value());
        break;
    default:
        ImShow = ShowTransparentRegionOnImage(TileMask, TileIm, ui->spinBoxTransparency->value());
        break;
    }
    ShowsScaledImage(ImShow,"Tile", ui->doubleSpinBoxTileScale->value());


    //ShowImages();
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::GetLesion()
{

    if(NoCommonLiesionRegions.empty())
    {
        destroyWindow("Lesion");
        return;
    }
    if(ReffMask.empty())
    {
        destroyWindow("Lesion");
        return;
    }
    if(LesionMask.empty())
    {
        destroyWindow("Lesion");
        return;
    }
    if(ImIn.empty())
    {
        destroyWindow("Lesion");
        return;
    }
    if(ui->spinBoxLesionNr->value() < 1)
    {
        destroyWindow("Lesion");
        return;
    }

    int lesionIndex = NoCommonLiesionRegions[ui->spinBoxLesionNr->value()-1];

    if(lesionIndex<=0)
    {
        destroyWindow("Lesion");
        return;
    }

    RegionParams LesionRegParams = LesionRegionsParams.GetRegionParams(lesionIndex);

    ui->checkBoxLesionValid->setChecked(LesionRegParams.valid);

    Mat TileIm, TileReffMask, TileLesionMask, TileMask;


    ImIn(Rect(LesionRegParams.minX,LesionRegParams.minY,LesionRegParams.sizeX,LesionRegParams.sizeY)).copyTo(TileIm);
    ReffMask(Rect(LesionRegParams.minX, LesionRegParams.minY, LesionRegParams.sizeX, LesionRegParams.sizeY)).copyTo(TileReffMask);
    LesionMask(Rect(LesionRegParams.minX, LesionRegParams.minY, LesionRegParams.sizeX, LesionRegParams.sizeY)).copyTo(TileLesionMask);
    Mask(Rect(LesionRegParams.minX, LesionRegParams.minY, LesionRegParams.sizeX, LesionRegParams.sizeY)).copyTo(TileMask);

    Mat ImShow;
    if(TileLesionMask.empty() || TileIm.empty())
        return;
    //TileIm.copyTo(ImShow);

    switch(1)//ui->comboBoxShowMode->currentIndex())
    {
    case 1:
        ImShow = ShowTransparentRegionOnImage(GetContour5(TileMask), TileIm, ui->spinBoxTransparency->value());
        break;
    default:
        ImShow = ShowTransparentRegionOnImage(TileMask, TileIm, ui->spinBoxTransparency->value());
        break;
    }
    ShowsScaledImage(ImShow,"Lesion", ui->doubleSpinBoxLesionScale->value());
    ui->spinBoxTileX->setValue(LesionRegParams.massCenterX - ui->spinBoxTileSize->value()/2);
    ui->spinBoxTileY->setValue(LesionRegParams.massCenterY - ui->spinBoxTileSize->value()/2);
    GetTile();
    //ShowImages();
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::GetReff()
{

    if(ReffLiesionRegions.empty())
    {
        destroyWindow("Reff");
        return;
    }
    if(ReffMask.empty())
    {
        destroyWindow("Reff");
        return;
    }
    if(LesionMask.empty())
    {
        destroyWindow("Reff");
        return;
    }
    if(ImIn.empty())
    {
        destroyWindow("Reff");
        return;
    }
    if(ui->spinBoxReffNr->value() < 1)
    {
        destroyWindow("Reff");
        return;
    }
    int lesionIndex = ReffLiesionRegions[ui->spinBoxReffNr->value()-1];

    if(lesionIndex<=0)
    {
        destroyWindow("Reff");
        return;
    }

    RegionParams LesionRegParams = RefRegionsParams.GetRegionParams(lesionIndex);



    Mat TileIm, TileReffMask, TileLesionMask, TileMask;


    ImIn(Rect(LesionRegParams.minX,LesionRegParams.minY,LesionRegParams.sizeX,LesionRegParams.sizeY)).copyTo(TileIm);
    ReffMask(Rect(LesionRegParams.minX, LesionRegParams.minY, LesionRegParams.sizeX, LesionRegParams.sizeY)).copyTo(TileReffMask);
    LesionMask(Rect(LesionRegParams.minX, LesionRegParams.minY, LesionRegParams.sizeX, LesionRegParams.sizeY)).copyTo(TileLesionMask);
    Mask(Rect(LesionRegParams.minX, LesionRegParams.minY, LesionRegParams.sizeX, LesionRegParams.sizeY)).copyTo(TileMask);

    Mat ImShow;
    if(TileLesionMask.empty() || TileIm.empty())
        return;
    //TileIm.copyTo(ImShow);

    switch(1)//ui->comboBoxShowMode->currentIndex())
    {
    case 1:
        ImShow = ShowTransparentRegionOnImage(GetContour5(TileMask), TileIm, ui->spinBoxTransparency->value());
        break;
    default:
        ImShow = ShowTransparentRegionOnImage(TileMask, TileIm, ui->spinBoxTransparency->value());
        break;
    }
    ShowsScaledImage(ImShow,"Reff", ui->doubleSpinBoxLesionScale->value());
    ui->spinBoxTileX->setValue(LesionRegParams.massCenterX - ui->spinBoxTileSize->value()/2);
    ui->spinBoxTileY->setValue(LesionRegParams.massCenterY - ui->spinBoxTileSize->value()/2);
    GetTile();
    //ShowImages();
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::AdiustTilePositionSpinboxes()
{
    int tileSizeX, tileSizeY;

    tileSizeX = ui->spinBoxTileSize->value();
    tileSizeY = ui->spinBoxTileSize->value();


    int imMaxX = ImIn.cols;
    int imMaxY = ImIn.rows;

    ui->spinBoxTileX->setMaximum(imMaxX-tileSizeX - 1);
    ui->spinBoxTileY->setMaximum(imMaxY-tileSizeY - 1);
    ui->spinBoxTileX->setSingleStep(tileSizeX/2);
    ui->spinBoxTileY->setSingleStep(tileSizeY/2);
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ScaleImMiniature()
{
    if(ImIn.empty())
        return;
    int scale = 10;//ui->spinBoxScaleBaseWhole->value();
    int scaledSizeX = ImIn.cols/scale;
    int scaledSizeY = ImIn.rows/scale;
    int positionX = 0;
    int positionY = 410;
    ui->widgetImageWhole->setGeometry(positionX,positionY,scaledSizeX,scaledSizeY);
}
//------------------------------------------------------------------------------------------------------------------------------
//                Slots
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::on_pushButtonOpenImageFolder_clicked()
{
    QFileDialog dialog(this, "Open Folder");
    //dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(ui->lineEditImageFolder->text());
    path ImageFolder;
    if(dialog.exec())
    {
        ImageFolder = dialog.directory().path().toStdWString();
    }
    else
        return;
    if (!exists(ImageFolder))
    {
        ui->textEditOut->append(" Image folder : " + QString::fromStdWString(ImageFolder.wstring())+ " not exists ");
        return;
    }
    if (!is_directory(ImageFolder))
    {
        ui->textEditOut->append(" Image folder : " + QString::fromStdWString(ImageFolder.wstring())+ " This is not a directory path ");
        return;
    }
    ui->lineEditImageFolder->setText(QString::fromStdWString(ImageFolder.wstring()));

    OpenImageFolder();
}

void MainWindow::on_listWidgetImageFiles_currentTextChanged(const QString &currentText)
{
    FileToOpen = ui->lineEditImageFolder->text().toStdWString();
    FileToOpen.append(currentText.toStdString());
    if(!exists(FileToOpen))
    {
        ui->textEditOut->append(" file : " + QString::fromStdWString(FileToOpen.wstring())+ " not exists ");
        return;
    }

    ReadImage();
}

void MainWindow::on_checkBoxShowInput_clicked()
{
    ShowImages();
}

void MainWindow::on_checkBoxShowTiffInfo_clicked()
{
    ShowImages();
}

void MainWindow::on_checkBoxShowMatInfo_clicked()
{
    ShowImages();
}

void MainWindow::on_checkBoxShowReffMask_clicked()
{
    ShowImages();
}

void MainWindow::on_checkBoxShowLesionMask_clicked()
{
    ShowImages();
}

void MainWindow::on_checkBoxShowLesionOnImage_clicked()
{
    ShowImages();
}

void MainWindow::on_checkBoxShowMaskAsContour_clicked()
{
    ShowImages();
}

void MainWindow::on_doubleSpinBoxImageScale_valueChanged(double arg1)
{
    ShowImages();
}

void MainWindow::on_widgetImageWhole_on_mousePressed(const QPoint point, int butPressed)
{
    int tileSize = ui->spinBoxTileSize->value();
    int tileHalfSize = tileSize /2;
    //int tileStep = ui->spinBoxTileStep->value();
    int maxXadjusted = ImIn.cols - tileSize - 1;
    int maxYadjusted = ImIn.cols - tileSize - 1;
    int miniatureScale = 10;//= ui->spinBoxScaleBaseWhole->value();
    int x = point.x() * miniatureScale - tileHalfSize;//((point.x() * miniatureScale - tileHalfSize) / tileStep) * tileStep;
    int y = point.y() * miniatureScale - tileHalfSize;//((point.y() * miniatureScale - tileHalfSize) / tileStep) * tileStep;
    if(x < 0)
        x = 0;
    if(x > maxXadjusted)
        x = maxXadjusted;
    if(y < 0)
        y = 0;
    if(y > maxYadjusted)
        y = maxYadjusted;
    allowMoveTile = 0;
    ui->spinBoxTileX->setValue(x);
    ui->spinBoxTileY->setValue(y);
    allowMoveTile = 1;
    GetTile();
    ShowImages();
}

void MainWindow::on_spinBoxLesionNr_valueChanged(int arg1)
{
    GetLesion();
    ShowImages();
}

void MainWindow::on_checkBoxLesionValid_clicked(bool checked)
{
    int lesionIndex = NoCommonLiesionRegions[ui->spinBoxLesionNr->value()-1];
    if(lesionIndex > 0)
    {
        LesionRegionsParams.SetValid(lesionIndex,checked);
    }

}

void MainWindow::on_pushButtonGetStatistics_clicked()
{



    int nonValidRegionCount = LesionRegionsParams.GetCountOfNonZeroAreaNonValid();

    ui->textEditOut->clear();
    ui->textEditOut->append("name\tRef#\tTP\tFN\tFP\tles#\tvalid not common#");
    ui->textEditOut->append(QString::fromStdString( FileToOpen.stem().string()) + "\t"
                            + QString::number(reffRegionCount) + "\t"
                            + QString::number(commonRegionCount) + "\t"
                            + QString::number(reffRegionCount - commonRegionCount)+ "\t"
                            + QString::number(nonValidRegionCount)+ "\t"
                            + QString::number(lesionRegionCount)+ "\t"
                            + QString::number(lesionRegionCount- nonValidRegionCount - commonRegionCount));

}

void MainWindow::on_doubleSpinBoxLesionScale_valueChanged(double arg1)
{
    GetLesion();
}

void MainWindow::on_checkBoxGrabKeyboard_toggled(bool checked)
{
    if(checked)
        ui->widgetImageWhole->grabKeyboard();
    else
        ui->widgetImageWhole->releaseKeyboard();
}

void MainWindow::on_widgetImageWhole_on_KeyPressed(int button)
{
    int lesionIndex = NoCommonLiesionRegions[ui->spinBoxLesionNr->value()-1];
    switch (button)
    {

    case Qt::Key_Down:
        ui->spinBoxLesionNr->setValue(ui->spinBoxLesionNr->value()+1);
        break;
    case Qt::Key_Up:
        ui->spinBoxLesionNr->setValue(ui->spinBoxLesionNr->value()-1);
        break;
    case Qt::Key_Home:
        ui->spinBoxLesionNr->setValue(1);
        break;

    case Qt::Key_Space:
        ui->checkBoxLesionValid->setChecked(false);
        if(lesionIndex > 0)
        {
            LesionRegionsParams.SetValid(lesionIndex,false);
        }
        break;
    case Qt::Key_X:
        ui->checkBoxLesionValid->setChecked(true);
        if(lesionIndex > 0)
        {
            LesionRegionsParams.SetValid(lesionIndex,true);
        }
        break;


    default:
        break;
    }
}

void MainWindow::on_pushButtonReload_clicked()
{
    ReadImage();
}

void MainWindow::on_pushButtonSaveOut_clicked()
{
    QString ImageFolderQStr;
    path outFilePath;
    ImageFolderQStr = ui->lineEditOutFolder->text();
    outFilePath = ImageFolderQStr.toStdWString();

    outFilePath.append(FileToOpen.stem().string() + ui->lineEditPostFix->text().toStdString() + ".bmp");
    Mat ImToSave = ShowSolidRegionOnImage(Mask, ImIn);
    imwrite(outFilePath.string(),ImToSave);

    ImageFolderQStr = ui->lineEditOutFolder->text();
    outFilePath = ImageFolderQStr.toStdWString();

    if(!ui->checkBox3Masks->checkState())
    {
        outFilePath.append(FileToOpen.stem().string() + ui->lineEditPostFix->text().toStdString() + "Tile.bmp");
        ImToSave = ShowSolidRegionOnImage(TileMask, TileIm);
        imwrite(outFilePath.string(),ImToSave);
    }
    else
    {
        outFilePath.append(FileToOpen.stem().string() + "Fussion" + "Tile.bmp");
        ImToSave = ShowSolidRegionOnImage(TileMask, TileIm);
        imwrite(outFilePath.string(),ImToSave);
    }

}

void MainWindow::on_pushButtonSaveQMaZdaStyleRoi_clicked()
{
    if(!ui->checkBox3Masks->checkState())
    {
        ui->textEditOut->append("this option is validd for 3 mask mode only");
        return;
    }
    if(LesionMask1.empty())
    {
        ui->textEditOut->append("MaKo data unavalible");
        return;
    }
    if(LesionMask2.empty())
    {
        ui->textEditOut->append("MaSt data unavalible");
        return;
    }
    if(LesionMask3.empty())
    {
        ui->textEditOut->append("MiKo data unavalible");
        return;
    }

    unsigned short *CommonRegSizes = new unsigned short[65536];

    for (unsigned int regNr = 0; regNr < 65536; regNr++)
    {
        CommonRegSizes[regNr] = 0;
    }

    int maxXY = CommonMask.cols * CommonMask.rows;

    uint16_t *wCommonMask = (uint16_t *)CommonMask.data;
    //unsigned short oldRegNr;
    for (int i = 0; i < maxXY; i++)
    {
        if(*wCommonMask)
            CommonRegSizes[*wCommonMask]++;
        wCommonMask++;
    }

    unsigned short *Exchange = new unsigned short[65536];
    for (unsigned int regNr = 0; regNr < 65536; regNr++)
    {
        Exchange[regNr] = 0;
    }

    unsigned short newRegNR = 1;
    for (unsigned int regNr = 0; regNr < 65536; regNr++)
    {
        if(CommonRegSizes[regNr] > 5)
        {
            Exchange[regNr] = newRegNR;
            newRegNR ++;
        }
    }

    Mat CombinedMaskReduced;
    CombinedMask.copyTo(CombinedMaskReduced);
    uint16_t *wCombinedMaskReduced = (uint16_t *)CombinedMaskReduced.data;
    for (int i = 0; i < maxXY; i++)
    {
        if(*wCombinedMaskReduced)
            *wCombinedMaskReduced = Exchange[*wCombinedMaskReduced];
        wCombinedMaskReduced++;
    }
    delete[] Exchange;
    delete[] CommonRegSizes;

//-------------------------To delete-START---------------------------
    Mat ImToShow;
    ShowSolidRegionOnImage(CombinedMaskReduced, ImIn).copyTo(ImToShow);
    double scale = 1 / ui->doubleSpinBoxImageScale->value();

    ShowsScaledImage(ImToShow,"Reduced Combined Mask on Image", scale);
//-------------------------To delete-END---------------------------

    RenumberMask(CombinedMaskReduced,LesionMask);
    RenumberMask(CombinedMaskReduced,LesionMask1);
    RenumberMask(CombinedMaskReduced,LesionMask2);
    RenumberMask(CombinedMaskReduced,LesionMask3);
    RenumberMask(CombinedMaskReduced,ReffMask);
    RenumberMask(CombinedMaskReduced,CommonMask);

    QString ImageFolderQStr = ui->lineEditImageFolder->text();
    path QMazdaRoiFilePath;

    QMazdaRoiFilePath = ImageFolderQStr.toStdWString();
    QMazdaRoiFilePath.append(FileToOpen.stem().string() + "Comb.roi");
    SaveQMaZdaROI(LesionMask, QMazdaRoiFilePath.string());

    QMazdaRoiFilePath = ImageFolderQStr.toStdWString();
    QMazdaRoiFilePath.append(FileToOpen.stem().string() + "MaKo.roi");
    SaveQMaZdaROI(LesionMask1, QMazdaRoiFilePath.string());

    QMazdaRoiFilePath = ImageFolderQStr.toStdWString();
    QMazdaRoiFilePath.append(FileToOpen.stem().string() + "MaSt.roi");
    SaveQMaZdaROI(LesionMask2, QMazdaRoiFilePath.string());

    QMazdaRoiFilePath = ImageFolderQStr.toStdWString();
    QMazdaRoiFilePath.append(FileToOpen.stem().string() + "MiKo.roi");
    SaveQMaZdaROI(LesionMask3, QMazdaRoiFilePath.string());

    QMazdaRoiFilePath = ImageFolderQStr.toStdWString();
    QMazdaRoiFilePath.append(FileToOpen.stem().string() + "Ref.roi");
    SaveQMaZdaROI(ReffMask, QMazdaRoiFilePath.string());
}

void MainWindow::on_pushButtonSaveReport_clicked()
{
     ui->textEditRaport->append(QString::fromStdWString(FileToOpen.stem().wstring()) + "\t" +
                            QString::number(aClassRefCount) + "\t" +
                            QString::number(aClassTPCount) + "\t" +
                            QString::number(aClassFNCount) + "\t" +
                            QString::number(bClassRefCount) + "\t" +
                            QString::number(bClassTPCount) + "\t" +
                            QString::number(bClassFNCount) + "\t" +
                            QString::number(cClassRefCount) + "\t" +
                            QString::number(cClassTPCount) + "\t" +
                            QString::number(cClassFNCount) + "\t" +
                            QString::number(totalRefCount) + "\t" +
                            QString::number(totalTPCount) + "\t" +
                            QString::number(totalFNCount) + "\t" +
                            QString::number(totalFPCount) + "\t" +
                            QString::number(LesionRegionsParams.GetCountOfNonZeroAreaNonValid()));
    ui->listWidgetImageFiles->setCurrentRow(ui->listWidgetImageFiles->currentRow()+1);
     /*
    ui->textEditRaport->append(QString::fromStdString( FileToOpen.stem().string()) + "\t"
                            + QString::number(reffRegionCount) + "\t"
                            + QString::number(commonRegionCount) + "\t"
                            + QString::number(reffRegionCount - commonRegionCount)+ "\t"
                            + QString::number(lesionRegionCount)+ "\t"
                            + QString::number(lesionRegionCount - commonRegionCount));
*/
}

void MainWindow::on_spinBoxReffNr_valueChanged(int arg1)
{
    GetReff();
    ShowImages();
}

void MainWindow::on_pushButtonCLRRep_clicked()
{
    ui->textEditRaport->clear();
    ui->textEditRaport->append(QString("File Name") + "\t" +
                               QString("aClassRefCount") + "\t" +
                               QString("aClassTPCount") + "\t" +
                               QString("aClassFNCount") + "\t" +
                               QString("bClassRefCount") + "\t" +
                               QString("bClassTPCount") + "\t" +
                               QString("bClassFNCount") + "\t" +
                               QString("cClassRefCount") + "\t" +
                               QString("cClassTPCount") + "\t" +
                               QString("cClassFNCount") + "\t" +
                               QString("totalRefCount") + "\t" +
                               QString("totalTPCount") + "\t" +
                               QString("totalFNCount") + "\t" +
                               QString("totalFPCount") + "\t" +
                               QString("totalFPNonValidCount") + "\t");

}

void MainWindow::on_pushButtonProcesFollowUp_clicked()
{
    Lesion1PosXVect.clear();
    Lesion1PosYVect.clear();
    Lesion2PosXVect.clear();
    Lesion2PosYVect.clear();

    path TextFile = ui->lineEditImageFolder->text().toStdWString();
    TextFile.append(FileToOpen.stem().string() + ".txt");

    if (!exists(TextFile))
    {
        ui->textEditOut->append("txt file does not exist");
        return;
    }
    std::ifstream inFile(TextFile.string());
    if (!inFile.is_open())
    {
        ui->textEditOut->append("txt file can not be opened");
        return;
    }
    string Line;
    while(inFile.good())
    {
        getline(inFile, Line);
        if(Line.empty())
            continue;
        double xPos1 =  round(stod( Line.substr(3, 15)));
        if(xPos1 < 0)
            xPos1 = 0;
        if(xPos1 > ImIn.cols)
            xPos1 = ImIn.cols;
        double yPos1 =  round(stod( Line.substr(20, 32)));
        if(yPos1 < 0)
            yPos1 = 0;
        if(yPos1 > ImIn.rows)
            yPos1 = ImIn.rows;
        double xPos2 =  round(stod( Line.substr(37, 49)));
        if(xPos2 < 0)
            xPos2 = 0;
        if(xPos2 > ImIn.cols)
            xPos2 = ImIn.cols;
        double yPos2 =  round(stod( Line.substr(54, 66)));
        if(yPos2 < 0)
            yPos2 = 0;
        if(yPos2 > ImIn.rows)
            yPos2 = ImIn.rows;

        ui->textEditOut->append(QString::number(xPos1) + "\t" + QString::number(yPos1) + "\t" + QString::number(xPos2) + "\t" + QString::number(yPos2));
        Lesion1PosXVect.push_back((int)xPos1);
        Lesion1PosYVect.push_back((int)yPos1);
        Lesion2PosXVect.push_back((int)xPos2);
        Lesion2PosYVect.push_back((int)yPos2);

    }
    inFile.close();

    if(Lesion1PosXVect.size())
    {
        ui->spinBoxFU->setMaximum(Lesion1PosXVect.size() - 1);
        ui->spinBoxFU->setEnabled(true);
    }
    else
    {
        ui->spinBoxFU->setMaximum(0);
        ui->spinBoxFU->setEnabled(false);
    }



}

void MainWindow::on_spinBoxFU_valueChanged(int arg1)
{
    ui->spinBoxTileX->setValue(Lesion1PosXVect[arg1] - ui->spinBoxTileSize->value()/2);
    ui->spinBoxTileY->setValue(Lesion1PosYVect[arg1] - ui->spinBoxTileSize->value()/2);
    GetTile();
    ShowImages();

}
