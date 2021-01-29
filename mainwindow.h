// ADRIAN TRUJILLO LOPEZ
// DNI : 21695691K

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <opennn/opennn.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void generateDataset();
    void leerCarpetaByN(std::string, std::string);
    void leerCarpetaBGR(std::string, std::string, size_t);
    void leerCarpetaHLS(std::string, std::string, size_t);
    void trainNetwork();
    void testNetwork();

private:

    OpenNN::DataSet data_set;
    OpenNN::LossIndex loss_index;
    OpenNN::TrainingStrategy training_strategy;
    OpenNN::TrainingStrategy::Results training_strategy_results;
    OpenNN::ModelSelection model_selection;
    OpenNN::ModelSelection::ModelSelectionResults model_selection_results;

    Ui::MainWindow *ui;
    std::string rutaDataset;
    bool Opcional = true;
    int line = 0;
};

#endif // MAINWINDOW_H
