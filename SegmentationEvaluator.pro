#-------------------------------------------------
#
# Project created by QtCreator 2020-10-19T20:25:03
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SegmentationEvaluator
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp\
        ../SkinLesionDiscovery/segmentMK.cpp \
        ../../ProjectsLib/LibMarcin/myimagewidget.cpp \
        ../../ProjectsLib/LibMarcin/NormalizationLib.cpp \
        ../../ProjectsLib/LibMarcin/DispLib.cpp \
        ../../ProjectsLib/LibMarcin/StringFcLib.cpp \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.cpp \
        ../../ProjectsLib/LibMarcin/histograms.cpp \
        ../../ProjectsLib/LibMarcin/gradient.cpp

HEADERS += \
        mainwindow.h\
        ../SkinLesionDiscovery/segmentMK.h \
        ../../ProjectsLib/LibMarcin/myimagewidget.h \
        ../../ProjectsLib/LibMarcin/NormalizationLib.h \
        ../../ProjectsLib/LibMarcin/DispLib.h \
        ../../ProjectsLib/LibMarcin/StringFcLib.h \
        ../../ProjectsLib/LibMarcin/RegionU16Lib.h \
        ../../ProjectsLib/LibMarcin/histograms.h \
        ../../ProjectsLib/LibMarcin/gradient.h


FORMS += \
        mainwindow.ui

win32: INCLUDEPATH += C:\opencv\build\include\
win32: INCLUDEPATH += C:\boost_1_66_0\
win32: INCLUDEPATH += ..\..\ProjectsLib\LibMarcin\
win32: INCLUDEPATH += ../SkinLesionDiscovery\
win32: INCLUDEPATH += C:\LibTiff\
win32: INCLUDEPATH += ../../ProjectsLibForein/LibPMS/
win32: INCLUDEPATH += ../../ProjectsLibForein/LibPMS/

# this is for debug
win32: LIBS += -LC:/opencv/build/x64/vc15/lib/
win32: LIBS += -lopencv_world341d

win32: LIBS += -LC:/boost_1_66_0/stage/x64/lib/
win32:  LIBS += -lboost_filesystem-vc141-mt-gd-x64-1_66
win32:  LIBS += -lboost_regex-vc141-mt-gd-x64-1_66


# this is for release
#win32: LIBS += -LC:/opencv/build/x64/vc15/lib/
#win32: LIBS += -lopencv_world341

#win32: LIBS += -LC:/boost_1_66_0/stage/x64/lib/
#win32: LIBS += -lboost_filesystem-vc141-mt-x64-1_66
#win32: LIBS += -lboost_regex-vc141-mt-x64-1_66




win32: LIBS += -LC:/LibTiff/
win32: LIBS += -llibtiff_i


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
