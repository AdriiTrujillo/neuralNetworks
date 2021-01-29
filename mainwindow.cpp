// ADRIAN TRUJILLO LÓPEZ
// DNI : 21695691K

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QFileDialog>
#include <QString>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

//Librerias OpenNN -------------------------
#include <opennn/opennn.h>
#include <tinyxml2/tinyxml2.h>
//------------------------------------------

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Connect button signal to appropriate slot
    connect(ui->createDataset, SIGNAL(released()), this, SLOT(generateDataset()));
    connect(ui->trainNet, SIGNAL(released()), this, SLOT(trainNetwork()));
    connect(ui->testNet, SIGNAL(released()), this, SLOT(testNetwork()));

    connect(this, SIGNAL(textChanged(QString)), this, SLOT(onTextChanged(QString)));
}

//_______________________________________________________________________________________________________________________________
//Funcion para generar el dataset
//_______________________________________________________________________________________________________________________________
void MainWindow::generateDataset()
{
    qDebug() << "Generate Dataset";
    QString datasetDir = QString(QFileDialog::getExistingDirectory(this, "Select dataset folder"));
    rutaDataset = datasetDir.toStdString() + "/dataset.csv";
    std::cout << rutaDataset<< std::endl;

    //_______________________________________________________________________________________________________________________________
    leerCarpetaByN(rutaDataset, "hombre");
    leerCarpetaByN(rutaDataset, "mujer");
    leerCarpetaByN(rutaDataset, "sin-persona");
    line = 0;
    //_______________________________________________________________________________________________________________________________
//    leerCarpetaBGR(rutaDataset, "hombre", 0);           //---
//    leerCarpetaBGR(rutaDataset, "mujer", 0);            //BGR
//    leerCarpetaBGR(rutaDataset, "sin-persona", 0);      //---
//    line = 0;
    //_______________________________________________________________________________________________________________________________
//    leerCarpetaHLS(rutaDataset, "hombre", 0);          //---
//    leerCarpetaHLS(rutaDataset, "mujer", 0);           //HLS
//    leerCarpetaHLS(rutaDataset, "sin-persona", 0);     //---
//    line = 0;
    //_______________________________________________________________________________________________________________________________

}

void MainWindow::trainNetwork()
{

    qDebug() << "Train Network";

    //srand((unsigned)time(NULL)); //Semilla para que los pesos de la red sean aleatorios

    //CARGAR EL DATASET ------------------------------------------------------------------

    data_set.set_data_file_name("../datasets_utilizados/dataset.csv");
    data_set.set_separator("Comma");
    std::cout << "Cargando el Dataset ... " << std::endl;
    data_set.load_data();
    std::cout << "Dataset cargado correctamente ... " << std::endl;

    //VARIABLES DEL DATASET --------------------------------------------------------------------------
    OpenNN::Variables* variables_pointer = data_set.get_variables_pointer();

    variables_pointer->set_use(2500, OpenNN::Variables::Target);
    variables_pointer->set_name(2500, "Hombres");
    variables_pointer->set_use(2501, OpenNN::Variables::Target);
    variables_pointer->set_name(2501, "Mujeres");
    variables_pointer->set_use(2502, OpenNN::Variables::Target);
    variables_pointer->set_name(2502, "Sin Persona");


    //Obtenemos información sobre las entradas y salidas
    std::cout << "Obteniendo información de la salida ... " << std::endl;
    OpenNN::Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
    OpenNN::Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

    std::cout << "DONE -  " << std::endl;

    //INSTANCIAS -------------------------------------------------------------------------

    OpenNN::Instances* instances_pointer = data_set.get_instances_pointer();

    std::cout << "Dividiendo el dataset ... " << std::endl;
    //Dividimos el dataset en tres partes (training, selection, test) (%)
    instances_pointer->split_random_indices(0.6, 0.2, 0.2);
    std::cout << "DONE - " << std::endl;

    //Escalar el dataset
    std::cout << "Escalando dataset ... " << std::endl;
    const OpenNN::Vector< OpenNN::Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
    const OpenNN::Vector< OpenNN::Statistics<double> > outputs_statistics = data_set.scale_targets_minimum_maximum();
    std::cout << "DONE - " << std::endl;

    //RED NEURONAL -----------------------------------------------------------------------

    //Configuramos la red nueronal con el numero de entradas que son los pixeles de la foto y 3 salidas que son las 3 clases

    std::cout << "Creando Red Neuronal ... " << std::endl;      
    OpenNN::Vector<size_t> capas(4);
    capas[0] = 2500;
    capas[1] = 15;
    capas[2] = 5;
    capas[3] = 3;
    OpenNN::NeuralNetwork neural_network(capas);

    OpenNN::MultilayerPerceptron* multilayer_perceptron_pointer = neural_network.get_multilayer_perceptron_pointer();

    //ELEGIR FUNCION DE ACTIVACION PARA LAS DIFERENTES CAPAS ---------------------------------------------------------------------
    //FUNCION HIPERBOLICA --------------------------------------------------------------------------------------------------------

    multilayer_perceptron_pointer->set_layer_activation_function(0, OpenNN::Perceptron::HyperbolicTangent);
    multilayer_perceptron_pointer->set_layer_activation_function(1, OpenNN::Perceptron::HyperbolicTangent);
//    multilayer_perceptron_pointer->set_layer_activation_function(2, OpenNN::Perceptron::HyperbolicTangent);
//    multilayer_perceptron_pointer->set_layer_activation_function(3, OpenNN::Perceptron::HyperbolicTangent);

    //FUNCION SIGMOIDE -----------------------------------------------------------------------------------------------------------

//    multilayer_perceptron_pointer->set_layer_activation_function(0, OpenNN::Perceptron::Logistic);
//    multilayer_perceptron_pointer->set_layer_activation_function(1, OpenNN::Perceptron::Logistic);
//    multilayer_perceptron_pointer->set_layer_activation_function(2, OpenNN::Perceptron::Logistic);
//    multilayer_perceptron_pointer->set_layer_activation_function(3, OpenNN::Perceptron::Logistic);

    //FUNCION RELU ---------------------------------------------------------------------------------------------------------------

//    multilayer_perceptron_pointer->set_layer_activation_function(0, OpenNN::Perceptron::Linear);
//    multilayer_perceptron_pointer->set_layer_activation_function(1, OpenNN::Perceptron::Linear);
    multilayer_perceptron_pointer->set_layer_activation_function(2, OpenNN::Perceptron::Linear);
//    multilayer_perceptron_pointer->set_layer_activation_function(3, OpenNN::Perceptron::Linear);
    //----------------------------------------------------------------------------------------------------------------------------
    std::cout << "DONE - " << std::endl;

    // ---------------------------------------------------------------------------------------------------------------------------
    // CONFIGURACIÓN DE LA RED ---------------------------------------------------------------------------------------------------

    std::cout << "Configurando información E/S ... " << std::endl;
    OpenNN::Inputs* inputs_pointer = neural_network.get_inputs_pointer();
    inputs_pointer->set_information(inputs_information);
    OpenNN::Outputs* outputs_pointer = neural_network.get_outputs_pointer();
    outputs_pointer->set_information(targets_information);
    std::cout << "DONE - " << std::endl;

    std::cout << "Creando Capa de Escalado ... " << std::endl;
    neural_network.construct_scaling_layer();
    OpenNN::ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
    scaling_layer_pointer->set_statistics(inputs_statistics);
    scaling_layer_pointer->set_scaling_method(OpenNN::ScalingLayer::NoScaling);
    std::cout << "DONE - " << std::endl;

    std::cout << "Creando Probabilidad ... " << std::endl;
    neural_network.construct_probabilistic_layer();
    OpenNN::ProbabilisticLayer* probability_layer_pointer = neural_network.get_probabilistic_layer_pointer();
    probability_layer_pointer->set_probabilistic_method(OpenNN::ProbabilisticLayer::Softmax);
    std::cout << "DONE - " << std::endl;

    // --------------------------------------------------------------------------------------------------------------------------
    //ENTRENAMIENTO -------------------------------------------------------------------------------------------------------------

    //ERROR DE LA RED -----------------------------------------------------------------------------------------------------------
    std::cout << "Creando el error de la red ... " << std::endl;
    loss_index.set_data_set_pointer(&data_set);
    loss_index.set_neural_network_pointer(&neural_network);

    //DIFERENTES ERRORES -------------------------------------------------------------------------------------------------------

    loss_index.set_error_type(OpenNN::LossIndex::NORMALIZED_SQUARED_ERROR);

//    loss_index.set_error_type(OpenNN::LossIndex::SUM_SQUARED_ERROR);
//    loss_index.set_error_type(OpenNN::LossIndex::MEAN_SQUARED_ERROR);
//    loss_index.set_error_type(OpenNN::LossIndex::ROOT_MEAN_SQUARED_ERROR);
//    loss_index.set_error_type(OpenNN::LossIndex::MINKOWSKI_ERROR);
//    loss_index.set_error_type(OpenNN::LossIndex::WEIGHTED_SQUARED_ERROR);
//    loss_index.set_error_type(OpenNN::LossIndex::ROC_AREA_ERROR);
//    loss_index.set_error_type(OpenNN::LossIndex::CROSS_ENTROPY_ERROR);
    //----------------------------------------------------------------------------------------------------------------------------

    loss_index.set_regularization_type(OpenNN::LossIndex::NO_REGULARIZATION);
    std::cout << "DONE - " << std::endl;

    std::cout << "Creando estrategia de entrenamiento ... " << std::endl;
    training_strategy.set_loss_index_pointer(&loss_index);

    //DIFERENTES ESTRATEGIAS ----------------------------------------------------------------------------------------------------
    //ESTRATEGIA : DESCENSO POR GRADIENTE ---------------------------------------------------------------------------------------
    training_strategy.set_main_type(OpenNN::TrainingStrategy::GRADIENT_DESCENT);
    OpenNN::GradientDescent* gradient_descent_pointer = training_strategy.get_gradient_descent_pointer();

    gradient_descent_pointer->set_maximum_iterations_number(21);
    gradient_descent_pointer->set_maximum_time(500000000); //Para que haga todas las iteraciones siempre
    gradient_descent_pointer->set_display_period(3);
    //---------------------------------------------------------------------------------------------------------------------------
    //ESTRATEGIA : CONJUGATE_GRADIENT -------------------------------------------------------------------------------------------
//    training_strategy.set_main_type(OpenNN::TrainingStrategy::CONJUGATE_GRADIENT);
//    OpenNN::ConjugateGradient* conjugate_gradient = training_strategy.get_conjugate_gradient_pointer();

//    conjugate_gradient->set_maximum_iterations_number(15);
//    conjugate_gradient->set_maximum_time(500000000); //Para que haga todas las iteraciones siempre
//    conjugate_gradient->set_display_period(3);
    //---------------------------------------------------------------------------------------------------------------------------
    //ESTRATEGIA : NEWTON_METHOD ------------------------------------------------------------------------------------------------
//    training_strategy.set_main_type(OpenNN::TrainingStrategy::NEWTON_METHOD);
//    OpenNN::NewtonMethod* newton_method = training_strategy.get_Newton_method_pointer();

//    newton_method->set_maximum_iterations_number(15);
//    newton_method->set_maximum_time(500000000); //Para que haga todas las iteraciones siempre
//    newton_method->set_display_period(3);
    //---------------------------------------------------------------------------------------------------------------------------
    //ESTRATEGIA : QUASI NEWTON_METHOD ------------------------------------------------------------------------------------------
//    training_strategy.set_main_type(OpenNN::TrainingStrategy::QUASI_NEWTON_METHOD);
//    OpenNN::QuasiNewtonMethod* quasi_newton_pointer = training_strategy.get_quasi_Newton_method_pointer();

//    quasi_newton_pointer->set_maximum_iterations_number(15);
//    quasi_newton_pointer->set_maximum_time(500000000); //Para que haga todas las iteraciones siempre
//    quasi_newton_pointer->set_display_period(3);
    //---------------------------------------------------------------------------------------------------------------------------
    //ESTRATEGIA : LEVENBERG_MARQUARDT_ALGORITHM --------------------------------------------------------------------------------
//    training_strategy.set_main_type(OpenNN::TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);
//    OpenNN::LevenbergMarquardtAlgorithm* levenberg_Marquardt_pointer = training_strategy.get_Levenberg_Marquardt_algorithm_pointer();

//    levenberg_Marquardt_pointer->set_maximum_iterations_number(15);
//    levenberg_Marquardt_pointer->set_maximum_time(500000000); //Para que haga todas las iteraciones siempre
//    levenberg_Marquardt_pointer->set_display_period(3);
    //---------------------------------------------------------------------------------------------------------------------------
    std::cout << "DONE - " << std::endl;

    //EMPIEZA EL ENTRENAMIENTO --------------------------------------------------------------------------------------------------
    std::cout << "Empieza el entrenamiento ... " << std::endl;
    training_strategy_results = training_strategy.perform_training();
    std::cout << "DONE - " << std::endl;
    //---------------------------------------------------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------------------------------------------------
    //MODEL SELECTION -----------------------------------------------------------------------------------------------------------
//    std::cout << "Creando Error de Modelo de selección ... " << std::endl;
//    model_selection.set_training_strategy_pointer(&training_strategy);
//    model_selection.set_inputs_selection_type(OpenNN::ModelSelection::GENETIC_ALGORITHM);
//    model_selection.set_order_selection_type(OpenNN::ModelSelection::INCREMENTAL_ORDER);

//    OpenNN::GeneticAlgorithm* genetic_algorithm_pointer = model_selection.get_genetic_algorithm_pointer();
//    genetic_algorithm_pointer->set_population_size(150);
//   //genetic_algorithm_pointer->set_tolerance(10e-7);

//    OpenNN::IncrementalOrder* incremental_order_pointer = model_selection.get_incremental_order_pointer();
//    //incremental_order_pointer->set_tolerance(1.0e-7);
//    std::cout << "DONE - " << std::endl;

//    std::cout << "Empieza el modelado ... " << std::endl;
//    model_selection_results = model_selection.perform_inputs_selection();
//    model_selection_results = model_selection.perform_order_selection();
//    std::cout << "Guardando datos ... " << std::endl;
//    model_selection.save("../data/model_selection.xml");
//    model_selection_results.save("../data/model_selection_results.dat");
//    std::cout << "DONE - " << std::endl;
    // --------------------------------------------------------------------------------------------------------------------------

    // --------------------------------------------------------------------------------------------------------------------------

    // GUARDAR CONFIGURACIÓN DE LA RED ------------------------------------------------------------------------------------------
    // ANALISIS DE TEST -------------------------------------------------------------

    OpenNN::TestingAnalysis testing_analysis(&neural_network, &data_set);

    OpenNN::Matrix<size_t> confusion = testing_analysis.calculate_confusion();

    std::cout << "Test terminado" << std::endl;
    std::cout << "La matriz de confusión es:" << std::endl;
    confusion.print();
    std::cout << std::endl;
    std::cout << "La red tiene una precisión de: " << (confusion.calculate_trace()/confusion.calculate_sum()*100)
              << "%" << std::endl;

    // GUARDAMOS LOS RESULTADOS ------------------------------------------------------------------

    data_set.save("../data/data_set.xml");

    neural_network.save("../data/neural_network.xml");
    neural_network.save_expression("../data/expression.txt");

    loss_index.save("../data/loss_index.xml");

    training_strategy.save("../data/training_strategy.xml");
    training_strategy_results.save("../data/training_strategy_results.dat");

    confusion.save("../data/confusion.dat");
}

void MainWindow::testNetwork()
{
    qDebug() << "Test Network";

    auto start = clock();

    const std::string xmlFile = "../redes_entrenadas/Red_entrenada.xml";
    OpenNN::NeuralNetwork neural_network(xmlFile);

    OpenNN::Vector<double> inputs(2500);
    OpenNN::Vector<double> outputs(3);
    size_t k = 0;

    const std::string ruta_image = "../dataset/imgs-test/itest-1.jpg";

    cv::Mat imagen = cv::imread(ruta_image);
    int anchura = imagen.cols; //Se guarda el tamaño para devolverla a su tamaño original al finalizar
    int altura = imagen.rows; //Se guarda el tamaño para devolverla a su tamaño original al finalizar
    cv::resize(imagen, imagen, cv::Size(250,250));
    cv::Mat grey;
    cv::cvtColor(imagen, grey, CV_BGR2GRAY); //Convertimos a escala de grises

    for(int i = 0; i <= grey.cols-50; i = i + 50){
        for(int j = 0; j <= grey.rows-50; j = j + 50){
            cv::Mat roi(grey, cv::Rect(i, j, 50, 50));
            k = 0;
            for( int i = 0; i < roi.rows; i++){
                for( int j = 0; j < roi.cols; j++){
                    inputs[k] = static_cast<int>(roi.at<uchar>(i,j));
                    k++;
                }
            }

            outputs = neural_network.calculate_outputs(inputs);
            long maxElementIndex = std::max_element(outputs.begin(),outputs.end()) - outputs.begin();

            if(maxElementIndex == 0){ //Hombres
                for (int  k = i; k < (roi.rows+i); k++) {
                  for (int  l = j; l < (roi.cols+j); l++) {
                    imagen.at<cv::Vec3b>(k,l)[0] = 255; //blue
                    imagen.at<cv::Vec3b>(k,l)[1] = 0;
                    imagen.at<cv::Vec3b>(k,l)[2] = 0;
                  }
                }
            }

            else if(maxElementIndex == 1){ //Mujeres
                for (int  k = i; k < (roi.rows+i); k++) {
                  for (int  l = j; l < (roi.cols+j); l++) {
                    imagen.at<cv::Vec3b>(k,l)[0] = 0;
                    imagen.at<cv::Vec3b>(k,l)[1] = 255; //green
                    imagen.at<cv::Vec3b>(k,l)[2] = 0;
                  }
                }
            }
        }
    }

    cv::imshow("RESULTADO", imagen);
    cv::resize(imagen, imagen, cv::Size(anchura,altura));
    cv::imwrite("../resultados/resultado1.jpg", imagen);

    auto end = clock();
    auto time = (1000.0 * (end-start) / CLOCKS_PER_SEC);

    std::cout << "Tiempo(ms): " << time << std::endl;
}

MainWindow::~MainWindow()
{
    delete ui;
}

//_______________________________________________________________________________________________________________________________
//Leer la carpeta con las imagenes en Blanco y Negro
//_______________________________________________________________________________________________________________________________
void MainWindow::leerCarpetaByN(std::string rutaDataset, std::string tipo){

    std::fstream fout;
    fout.open(rutaDataset, std::ios::out | std::ios::app);

    if(tipo.compare("hombre") == 0){ //Pertenecen a la clase 0

        std::vector<cv::String> rutas_Imagenes;
        //Leemos la ruta de todos los archivos terminados en .jpg y las guardamos en un vector
        cv::glob("../dataset/hombre/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
            line++;
            cv::Mat imagen = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
            cv::resize(imagen, imagen, cv::Size(50,50));

            for( int i = 0; i < imagen.rows; i++){
                for( int j = 0; j < imagen.cols; j++){
                    fout << static_cast<int>(imagen.at<uchar>(i,j)) << ",";

                }
            }

            //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
            (line != 303) ?  fout << "1,0,0" << std::endl : fout << "1,0,0"; //1,0,0 = Hombres;
        }

        fout.close(); //Cerramos el archivo de escritura
    }

    if(tipo.compare("mujer") == 0){ //Pertenecen a la clase 1

        std::vector<cv::String> rutas_Imagenes;
        //Leemos la ruta de todos los archivos terminados en .jpg y las guardamos en un vector
        cv::glob("../dataset/mujer/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
            line++;
            cv::Mat imagen = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
            cv::resize(imagen, imagen, cv::Size(50,50));

            for( int i = 0; i < imagen.rows; i++){
                for( int j = 0; j < imagen.cols; j++){
                    fout << static_cast<int>(imagen.at<uchar>(i,j)) << ",";
                }
            }

            (line != 303) ?  fout << "0,1,0" << std::endl : fout << "0,1,0"; //0,1,0 = Mujeres;
        }

        fout.close(); //Cerramos el archivo
    }

    if(tipo.compare("sin-persona") == 0){ //Pertenecen a la clase 2

        std::vector<cv::String> rutas_Imagenes;
        //Leemos la ruta de todos los archivos terminados en .jpg y las guardamos en un vector
        cv::glob("../dataset/sin-persona/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
            line++;
            cv::Mat imagen = cv::imread(ruta, cv::IMREAD_GRAYSCALE);
            cv::resize(imagen, imagen, cv::Size(50,50));

            for( int i = 0; i < imagen.rows; i++){
                for( int j = 0; j < imagen.cols; j++){
                    fout << static_cast<int>(imagen.at<uchar>(i,j)) << ",";
                }
            }

            (line != 303) ?  fout << "0,0,1" << std::endl : fout << "0,0,1"; //0,0,1 = sin-persona;
        }

        fout.close(); //Cerramos el archivo
    }


}

//_______________________________________________________________________________________________________________________________
//El parametro color elige uno de los canales RGB;
//los canales se dividen como canales[0] ->Blue, canales[1] --> Green, canales[2]-->Red
//Porque realmente los canales en OpenCV son BGR y no RGB
//_______________________________________________________________________________________________________________________________
void MainWindow::leerCarpetaBGR(std::string rutaDataset, std::string tipo, size_t color){

    std::fstream fout;
    fout.open(rutaDataset, std::ios::out | std::ios::app);

    if(tipo.compare("hombre") == 0){ //Pertenecen a la clase 0

        std::vector<cv::String> rutas_Imagenes;
        cv::glob("../dataset/hombre/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
            line++;
            cv::Mat imagen = cv::imread(ruta);
            cv::resize(imagen, imagen, cv::Size(50,50));
            std::vector<cv::Mat> canales;
            cv::split(imagen, canales); //Cada canal es una matriz de 8 bits
               //los canales se dividen como canales[0] ->Blue, canales[1] --> Green, canales[2]-->Red

            for( int i = 0; i < imagen.rows; i++){
                for( int j = 0; j < imagen.cols; j++){
                    fout << static_cast<int>(canales[color].at<uchar>(i,j)) << ",";
                }
            }

            //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
            (line != 303) ?  fout << "1,0,0" << std::endl : fout << "1,0,0"; //1,0,0 = Hombres;
        }
        fout.close(); //Cerramos el archivo
    }

    if(tipo.compare("mujer") == 0){ //Pertenecen a la clase 1

        std::vector<cv::String> rutas_Imagenes;
        cv::glob("../dataset/mujer/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
           line++;
           cv::Mat imagen = cv::imread(ruta);
           cv::resize(imagen, imagen, cv::Size(50,50));
           std::vector<cv::Mat> canales;
           cv::split(imagen, canales); //Cada canal es una matriz de 8 bits
           //los canales se dividen como canales[0] ->Blue, canales[1] --> Green, canales[2]-->Red

           for( int i = 0; i < imagen.rows; i++){
               for( int j = 0; j < imagen.cols; j++){
                   fout << static_cast<int>(canales[color].at<uchar>(i,j)) << ",";
               }
           }

           //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
           (line != 303) ?  fout << "0,1,0" << std::endl : fout << "0,1,0"; //0,1,0 = Mujeres;
        }
        fout.close(); //Cerramos el archivo
    }

    if(tipo.compare("sin-persona") == 0){ //Pertenecen a la clase 2

        std::vector<cv::String> rutas_Imagenes;
        cv::glob("../dataset/sin-persona/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
           line++;
           cv::Mat imagen = cv::imread(ruta);
           cv::resize(imagen, imagen, cv::Size(50,50));
           std::vector<cv::Mat> canales;
           cv::split(imagen, canales); //Cada canal es una matriz de 8 bits (Escala de grises)
           //los canales se dividen como canales[0] ->Blue, canales[1] --> Green, canales[2]-->Red

           for( int i = 0; i < imagen.rows; i++){
               for( int j = 0; j < imagen.cols; j++){
                   fout << static_cast<int>(canales[color].at<uchar>(i,j)) << ",";
               }
           }

           //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
           (line != 303) ?  fout << "0,0,1" << std::endl : fout << "0,0,1"; //0,0,1 = Sin-persona;
        }
        fout.close(); //Cerramos el archivo
    }

}

//_______________________________________________________________________________________________________________________________
//El parametro color elige uno de los canales HSL;
//los canales se dividen como canales[0] ->Matiz(Hue), canales[1] --> Lightness(Luminosidad), canales[2]-->Saturation(Saturacion)
//Porque realmente los canales en OpenCV son HLS y no HSL
//_______________________________________________________________________________________________________________________________
void MainWindow::leerCarpetaHLS(std::string rutaDataset, std::string tipo, size_t color){

    std::fstream fout;
    fout.open(rutaDataset, std::ios::out | std::ios::app);

    if(tipo.compare("hombre") == 0){ //Pertenecen a la clase 0

        std::vector<cv::String> rutas_Imagenes;
        cv::glob("../dataset/hombre/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
           line++;
           cv::Mat imagen = cv::imread(ruta);
           cv::resize(imagen, imagen, cv::Size(50,50));
           cv::Mat converted;
           cv::cvtColor(imagen, converted, CV_BGR2HLS); //Convertimos a Espacio de color HLS

           std::vector<cv::Mat> canales;
           cv::split(converted, canales); //Cada canal es una matriz de 8 bits
           //canales[0] ->Matiz(Hue), canales[1] --> Lightness(Luminosidad), canales[2]-->Saturation(Saturacion)

           for( int i = 0; i < converted.rows; i++){
               for( int j = 0; j < converted.cols; j++){
                   fout << static_cast<int>(canales[color].at<uchar>(i,j)) << ",";
               }
           }

           //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
           (line != 303) ?  fout << "1,0,0" << std::endl : fout << "1,0,0"; //1,0,0 = Hombres;
        }
        fout.close(); //Cerramos el archivo
    }

    if(tipo.compare("mujer") == 0){ //Pertenecen a la clase 1

        std::vector<cv::String> rutas_Imagenes;
        cv::glob("../dataset/mujer/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
           line++;
           cv::Mat imagen = cv::imread(ruta);
           cv::resize(imagen, imagen, cv::Size(50,50));
           cv::Mat converted;
           cv::cvtColor(imagen, converted, CV_BGR2HLS); //Convertimos a Espacio de color HLS

           std::vector<cv::Mat> canales;
           cv::split(converted, canales); //Cada canal es una matriz de 8 bits
           //canales[0] ->Matiz(Hue), canales[1] --> Lightness(Luminosidad), canales[2]-->Saturation(Saturacion)

           for( int i = 0; i < converted.rows; i++){
               for( int j = 0; j < converted.cols; j++){
                   fout << static_cast<int>(canales[color].at<uchar>(i,j)) << ",";
               }
           }

           //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
           (line != 303) ?  fout << "0,1,0" << std::endl : fout << "0,1,0"; //0,1,0 = Mujeres;
        }
        fout.close(); //cerramos el archivo
    }

    if(tipo.compare("sin-persona") == 0){ //Pertenecen a la clase 2

        std::vector<cv::String> rutas_Imagenes;
        cv::glob("../dataset/sin-persona/*.jpg", rutas_Imagenes);

        for(auto ruta : rutas_Imagenes){
           line++;
           cv::Mat imagen = cv::imread(ruta);
           cv::resize(imagen, imagen, cv::Size(50,50));
           cv::Mat converted;
           cv::cvtColor(imagen, converted, CV_BGR2HLS); //Convertimos a Espacio de color HLS

           std::vector<cv::Mat> canales;
           cv::split(converted, canales); //Cada canal es una matriz de 8 bits
           //canales[0] ->Matiz(Hue), canales[1] --> Lightness(Luminosidad), canales[2]-->Saturation(Saturacion)

           for( int i = 0; i < converted.rows; i++){
               for( int j = 0; j < converted.cols; j++){
                   fout << static_cast<int>(canales[color].at<uchar>(i,j)) << ",";
               }
           }

           //Necesitamos que hayan 3 columnas para las salidas porque tenemos 3 clases.
           (line != 303) ?  fout << "0,0,1" << std::endl : fout << "0,0,1"; //0,0,1 = Sin-persona;
        }
        fout.close(); //cerramos el archivo
    }

}
