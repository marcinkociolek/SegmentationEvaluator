#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "RegionU16Lib.h"

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButtonOpenImageFolder_clicked();

    void on_listWidgetImageFiles_currentTextChanged(const QString &currentText);

    void on_checkBoxShowInput_clicked();

    void on_checkBoxShowTiffInfo_clicked();

    void on_checkBoxShowMatInfo_clicked();

    void on_checkBoxShowReffMask_clicked();

    void on_checkBoxShowLesionMask_clicked();

    void on_checkBoxShowLesionOnImage_clicked();

    void on_checkBoxShowMaskAsContour_clicked();

    void on_doubleSpinBoxImageScale_valueChanged(double arg1);

    void on_widgetImageWhole_on_mousePressed(const QPoint point, int butPressed);

    void on_spinBoxLesionNr_valueChanged(int arg1);

    void on_checkBoxLesionValid_clicked(bool checked);

    void on_pushButtonGetStatistics_clicked();

    void on_doubleSpinBoxLesionScale_valueChanged(double arg1);

    void on_checkBoxGrabKeyboard_toggled(bool checked);

    void on_widgetImageWhole_on_KeyPressed(int );

    void on_pushButtonReload_clicked();

    void on_pushButtonSaveOut_clicked();

private:
    Ui::MainWindow *ui;

    boost::filesystem::path FileToOpen;
    //boost::filesystem::path FileName;
    //std::string FileNameTxt;
    cv::Mat ImIn;
    cv::Mat ImOut;

    cv::Mat LesionMask1;
    cv::Mat LesionMask2;
    cv::Mat LesionMask3;

    cv::Mat LesionMask;
    cv::Mat LesionMaskWithoutCommon;
    cv::Mat ReffMask;
    cv::Mat Mask;
    cv::Mat CommonMask;

    double minIm;
    double maxIm;

    bool allowMoveTile;
    bool adwancedMode;

    MultiRegionsParams LesionRegionsParams;
    MultiRegionsParams CommonRegionsParams;
    MultiRegionsParams RefRegionsParams;

    int reffRegionCount;
    int lesionRegionCount;
    int commonRegionCount;

    //MultiRegionsParams LesionRegionsParams;

    std::vector<int>NoCommonLiesionRegions;

    void OpenImageFolder();
    void ReadImage();
    void ShowsScaledImage(cv::Mat Im, std::string ImWindowName, double dispalyScale);
    void ShowImages();
    void LoadReffMask();
    void MaskFusion();
    void LoadLesionMask();
    cv::Mat LoadLesionMask(std::string PostFix);
    void ProcessImages();
    void GetTile();
    void GetLesion();
    void AdiustTilePositionSpinboxes();
    void ScaleImMiniature();
};

#endif // MAINWINDOW_H
