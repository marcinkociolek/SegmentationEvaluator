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

#include "mazdaroi.h"
#include "mazdaroiio.h"

#include <tiffio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace boost;
using namespace std;
using namespace boost::filesystem;
using namespace cv;


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
    LesionMask1 =  LoadLesionMask("MaKo");
    LesionMask2 =  LoadLesionMask("MaSt");
    LesionMask3 =  LoadLesionMask("MiKo");

    LesionMask = Combine3Regions(LesionMask1, LesionMask2, LesionMask3);
    int CombinedRegionCount = DivideSeparateRegions(LesionMask, 10);
    RenumberMask(LesionMask, LesionMask1);
    RenumberMask(LesionMask, LesionMask2);
    RenumberMask(LesionMask, LesionMask3);

    MultiRegionsParams LesionMask1Params(LesionMask1);
    MultiRegionsParams LesionMask2Params(LesionMask2);
    MultiRegionsParams LesionMask3Params(LesionMask3);

    for(unsigned short k = 1; k <= CombinedRegionCount; k++)
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
        ReffMask = imread(maskFilePath.string(), CV_LOAD_IMAGE_ANYDEPTH);
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
    maskFilePath.append(FileToOpen.stem().string() + ui->lineEditPostFix->text().toStdString() + ".bmp");

    if(!exists(maskFilePath))
    {
        ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return;
    }

    Mat LesionMaskTemp;
    LesionMask = Mat::zeros(ImIn.rows, ImIn.cols, CV_16U);
    if(exists(maskFilePath))
    {
        LesionMaskTemp = imread(maskFilePath.string(), CV_LOAD_IMAGE_ANYDEPTH);
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
        if(*wLesionMaskTemp == 255)
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
    maskFilePath.append(FileToOpen.stem().string() + PostFix+ ".bmp");

    if(!exists(maskFilePath))
    {
        ui->textEditOut->append("lesion mask file " + QString::fromStdWString(maskFilePath.wstring()) + " not exists");
        return LesionMaskOut;
    }


    Mat LesionMaskTemp;
    if(exists(maskFilePath))
    {
        LesionMaskTemp = imread(maskFilePath.string(), CV_LOAD_IMAGE_ANYDEPTH);
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
        if(*wLesionMaskTemp == 255)
            *wLesionMaskOut = 1;
        wLesionMaskOut++;
        wLesionMaskTemp++;
    }
    return LesionMaskOut;

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
        ShowsScaledImage(ImToShow, "Mask", scale);
    }


    if(ui->checkBoxShowLesionMask->checkState())
    {
        Mat ImToShow;
        ShowRegion(LesionMask).copyTo(ImToShow);
        rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);
        ShowsScaledImage(ImToShow, "Lesion mask", scale);

    }
    if(ui->checkBoxShowLesionOnImage->checkState())
    {
        Mat ImToShow;
        if(ui->checkBoxShowMaskAsContour->checkState())
            ShowSolidRegionOnImage(GetContour5(Mask), ImIn).copyTo(ImToShow);
        else
            ShowSolidRegionOnImage(Mask, ImIn).copyTo(ImToShow);

        rectangle(ImToShow, Rect(tilePositionX,tilePositionY, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), lineThickness);

        int lesionIndex = NoCommonLiesionRegions[ui->spinBoxLesionNr->value()-1];

        if(lesionIndex > 0)
        {
            RegionParams LesionRegParams = LesionRegionsParams.GetRegionParams(lesionIndex);

            rectangle(ImToShow, Rect(LesionRegParams.minX,
                                     LesionRegParams.minY,
                                     LesionRegParams.sizeX,
                                     LesionRegParams.sizeY),
                      Scalar(0.0, 0.0, 255.0, 0.0),
                      -1);
        }
        ShowsScaledImage(ImToShow,"Lesion mask on Image", scale);
    }

    if(ui->checkBoxShowLesion->checkState())
    {
        //Mat LesionImToShow;


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

    Mat CombinedMask = Combine2Regions(ReffMask, LesionMask);

    if(CombinedMask.empty())
        return;

    int combinedMaskCount = DivideSeparateRegions(CombinedMask, 10);
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
    LesionRegionsParams.GetFromMat(LesionMask);
    //LesionRegionsParams();
    CommonRegionsParams.GetFromMat(CommonMask);
    RefRegionsParams.GetFromMat(ReffMask);

    MultiRegionsParams CombinedRegionParams(CombinedMask);

    reffRegionCount = RefRegionsParams.GetCountOfNonZeroArea();
    lesionRegionCount = LesionRegionsParams.GetCountOfNonZeroArea();
    commonRegionCount = CommonRegionsParams.GetCountOfNonZeroArea();
    int combinedRegionCount = CombinedRegionParams.GetCountOfNonZeroArea();
    ui->textEditOut->append("Combined region count = " + QString::number(combinedMaskCount));
    ui->textEditOut->append("Combined region count = " + QString::number(combinedRegionCount));
    ui->textEditOut->append("Refference region count = " + QString::number(reffRegionCount));
    ui->textEditOut->append("Lession region count = " + QString::number(lesionRegionCount));
    ui->textEditOut->append("Common region count = " + QString::number(commonRegionCount));

    NoCommonLiesionRegions.clear();
    for(int k = 1; k <= combinedMaskCount; k++)
    {
        RegionParams LesionRegParam = LesionRegionsParams.GetRegionParams(k);
        RegionParams CommonRegParam = CommonRegionsParams.GetRegionParams(k);
        if(LesionRegParam.area && (CommonRegParam.area == 0))
            NoCommonLiesionRegions.push_back(k);
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

/*
    reffRegionCount = DivideSeparateRegions(ReffMask, 10);
    lesionRegionCount = DivideSeparateRegions(LesionMask, 10);
    ui->textEditOut->append("Refference region count = " + QString::number(reffRegionCount));
    ui->textEditOut->append("Lession region count = " + QString::number(lesionRegionCount));

    int maxX = ReffMask.cols;
    int maxY = ReffMask.rows;
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

    commonRegionCount = DivideSeparateRegions(CommonMask, 0);
    ui->textEditOut->append("Common region count = " + QString::number(commonRegionCount));

    wReffMask = (unsigned short *)ReffMask.data;
    wLesionMask = (unsigned short *)LesionMask.data;
    wMask = (unsigned short *)Mask.data;
    wCommonMask = (unsigned short *)CommonMask.data;

    delete [] LesionRegionsParams;
    LesionRegionsParams = nullptr;

    if(lesionRegionCount)
    {
        LesionRegionsParams = new RegionParams[lesionRegionCount + 1];
        for(int k = 0; k <= lesionRegionCount; k++)
        {
            LesionRegionsParams[k].Init();
        }
        for(int y = 0; y < maxY; y++)
        {
            for(int x = 0; x < maxX; x++)
            {
                int regionIndex = *wLesionMask;
                LesionRegionsParams[regionIndex].area++;
                if(LesionRegionsParams[regionIndex].maxX < x)
                    LesionRegionsParams[regionIndex].maxX = x;
                if(LesionRegionsParams[regionIndex].maxY < y)
                    LesionRegionsParams[regionIndex].maxY = y;
                if(LesionRegionsParams[regionIndex].minX > x)
                    LesionRegionsParams[regionIndex].minX = x;
                if(LesionRegionsParams[regionIndex].minY > y)
                    LesionRegionsParams[regionIndex].minY = y;
                LesionRegionsParams[regionIndex].massCenterX += x;
                LesionRegionsParams[regionIndex].massCenterY += y;

                wLesionMask++;
            }
        }


        ui->spinBoxLesionNr->setMinimum(1);
        ui->spinBoxLesionNr->setMaximum(lesionRegionCount);
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
*/
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

    Mat TileIm, TileReffMask, TileLesionMask, TileMask;

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

    case Qt::Key_Up:
        ui->spinBoxLesionNr->setValue(ui->spinBoxLesionNr->value()+1);
        break;
    case Qt::Key_Down:
        ui->spinBoxLesionNr->setValue(ui->spinBoxLesionNr->value()-1);
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
    QString ImageFolderQStr = ui->lineEditImageFolder->text();
    path outFilePath = ImageFolderQStr.toStdWString();

    outFilePath.append(FileToOpen.stem().string() + ui->lineEditPostFix->text().toStdString() + ".jpg");
    Mat ImToSave = ShowSolidRegionOnImage(Mask, ImIn);
    imwrite(outFilePath.string(),ImToSave);
}
