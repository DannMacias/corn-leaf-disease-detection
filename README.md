# <b> Disease detection in maize crops using image classification </b> 🌽🚀

## Introduction
The proliferation of pathogens in crops is a major cause of losses for farmers. Early identification of infected plants prevents crop losses.
Corn is a crop of commercial interest, especially in Mexico, and is affected by a wide variety of pathogens:
* **Gray leaf spot** (*Cercospora zeae-maydis*) is considered the greatest threat to corn crops in many parts of the world, causing losses of up to 100% in infested fields. In the United States alone, it has affected more than 20 million acres; in Mexico, it has caused losses in the northwestern regions of the country, particularly in Sonora and Sinaloa. The visible symptoms of the disease are lesions in the form of necrotic spots that vary in color from gray to brown.
* **Corn leaf blight** (*Helminthosporium turcicum*) is a fungus that attacks only the leaves, preventing photosynthesis and reducing the size of the corn crop. The first symptoms are the appearance of grayish-green spots that later become brown and elongated, up to 6 inches (15 cm) long.
* **Common rust** (*Puccinia sorghi*) is one of the most common diseases of corn. The characteristic symptoms of this disease are orange-brown pustules covering both sides of the leaf.

In most cases, applying fungicides, rotating crops or using genetically modified corn are effective ways to prevent these pests from developing. However, early detection is the first step in controlling these diseases.

Significant advances in image processing have been achieved with deep learning techniques. Many applications have been developed for automatic identification of plant diseases.

These applications can be used to increase crop production by timely detection of areas affected by a pathogen.
In this project, different deep learning models were tested to detect corn leaf diseases: gray leaf spot, common rust, and blight. In addition to deploying the model using the Gradio interface. 

## Methods

The database used was Corn Maize Leaf Disease, which contains 4188 images of corn plants with leaf diseases. The description of the dataset is:
* 0 - Common Rust (1306 images).
* 1 - Gray Leaf Spot (574 images).
* 2 - Blight (1146 images).
* 3 - Healthy (1162 images).

This dataset has been made using the popular [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) and [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) datasets, and is available in [Kaggle](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset). 

Different data augmentation techniques were evaluated:
* Color Jitter.
* Gaussian Blur.
* Random Invert.
* TrivialAugmentWide.
* AugMix.

The six transformations plus a control (no data augmentation) were compared. In this first experiment, the model used was EfficientNetB2. The transformation with the best performance (accuracy better than 90%) was selected. 

The performance of five models was evaluated:
* EfficientNetB2 (model base or control).
* AlexNet.
* GoogleNet.
* ResNet50
* ViT.

The model that had an accuracy of more than 90% on the test set, as well as a latency of less than 0.3 seconds, was selected for deployment. The two models that met these conditions were GoogleNet and ResNet50. Both models were compared under the following conditions:
* A different training time: 5 epochs vs. 10 epochs.
* A different training set: with data augmentation vs. No data augmentation.
* A different model: GoogleNet vs. ResNet50.

The results were tracked using TensorBoard. Finally, the best performing model was deployed using a Gradio interface.

## Results

### First model: EfficientNetB2 

A first model evaluated was EfficientNetB2, with a test set accuracy of 90%. The predictions with error came from Gray Leaf Spot. This may be due to the small number of images in this class (500 images, half compared to the other classes).

![](https://github.com/DannMacias/corn-leaf-disease-detection/blob/main/images/confusion-matix-effnetb2.png)

### Data augmentation experiment

When comparing the different data augmentation techniques, TrivialAugmentWide was the one that improved the accuracy of the test set, followed by RandomAugment and AugMix. The worst performing technique was Gaussian Blur, with an accuracy of less than 85% and the highest loss values.

![](https://github.com/DannMacias/corn-leaf-disease-detection/blob/main/images/data-augmentation-results.png)

TrivialAugmentWide changes the colors of the images to help the model better distinguish between the different classes. Since the difference between Blight and Gray Leaf Spot, like Common Rust, is very small, these transformations help the model to better discriminate between the different classes.

![](https://github.com/DannMacias/corn-leaf-disease-detection/blob/main/images/data-augmentation-images.png)

### Comparison of different models

In each experiment, the conditions are the same: train test without data augmentation, 10 epochs, a batch size of 32, CrossEntropyLoss as criterion and Adam with a learning rate of 0.001 as optimizer.

To compare the models, the average time per prediction (on the test set images) and the last value of the test set accuracy were calculated. The results are shown in the scatter plot below, which also shows the size of the model (measured in megabytes).

![](https://github.com/DannMacias/corn-leaf-disease-detection/blob/main/images/corn-disease-detection-inference-speed-vs-performance.png)

Of the models evaluated, ViT has the highest accuracy. However, this comes at the cost of a larger size and a longer time to make predictions. If we want to use the model, especially on mobile devices, the size of ViT is a major limitation.

ResNet50 and AlexNet have the same test accuracy, but ResNet50 is smaller than AlexNet (2.4 times smaller). Currently, ResNet50 seems to be the model of choice, followed by GoogleNet. ResNet50 has the same test accuracy value as AlexNet, has a smaller size, but takes longer to infer (1.6 times slower than GoogleNet).

GoogleNet has the smallest size, it is fast to infer, but its accuracy is 91%, which is the second worst accuracy, only beaten by EffNetB2.

Since we are interested in being able to deploy the model, the size of the model is a factor to consider. Therefore, we chose the GoogleNet and ResNet50 models because they have a smaller size, in addition to being fast in inference and having an accuracy greater than 91%.

### Experiment tracking with TensorBoard: GoogleNet vs. ResNet50 and model deployment

A comparison between the GoogleNet and ResNet50 models is then performed. The following combinations were tested: different training time (5 vs. 10 epochs), different training set (with TrivialAugmentWide data augmentation and without data augmentation).

|**Experiment number**|**Training Dataset**|**Model**|**Number of epochs**|
|-|-|-|-|
|1|No data augmentation|GoogleNet|5|
|2|No data augmentation|ResNet50|5|
|3|No data augmentation|GoogleNet|10|
|4|No data augmentation|ResNet50|10|
|5|Data augmentation|GoogleNet|5|
|6|Data augmentation|ResNet50|5|
|7|Data augmentation|GoogleNet|10|
|8|Data augmentation|ResNet50|10|

TensorBoard, a visualization program developed by the TensorFlow team, was used to monitor the experiments. The results can be seen at the following [**TensorBoard link**](https://tensorboard.dev/experiment/nJow8Wg7QqWinqxvkxJXcg/#scalars).

Looking at the TensorBoard logs for the eight experiments, the experiment number 4 achieved the best overall results (high test accuracy and lowest test loss). This experiment consist in a ResNet50 model, no data augmentation and a train time of 10 epochs.

The second best experiment was the ResNet50 model, 10 epochs and data augmentation. Based on these results, the best model is ResNet50, without data augmentation and with a training time of 10 epochs. The results are shown in the following graphic.

![](https://github.com/DannMacias/corn-leaf-disease-detection/blob/main/images/resnet50-results.png)

These are the conditions that will be used to deploy the model in the Gradio interface. The Gradio interface is shown below.

![](https://github.com/DannMacias/corn-leaf-disease-detection/blob/main/images/gradio-interface.png)

## Conclusions


## References
1. Arun, J., and Gopal, G. (2019). Data for: Identification of plant leaf disease using a 9-layer deep convolutional neural network. Mendeley data, V1.
2. Boulent, J., Foucher, S., Théau, J., and Charles P. (2019). Convolutional neural network for the automatic identification of plant diseases. *Frontiers in Plant Science*, 10:941.
3. R. (2022, 23 junio). Enfermedades foliares del maíz. Revista InfoAgro México, https://mexico.infoagro.com/enfermedades-foliares-del-maiz/#:~:text=Cercospora%20zeae%2Dmaydis%20se%20desarrolla,Stromberg%20y%20Donahue%2C%201986).
4. Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., and Batra, N. (2020). PlantDoc: a dataset for visual plant disease detection. In Proceedings of the 7th ACM IKDD CoDS and 25th COMAD 2020 Jan 5 (pp. 249-253).
5. Ward, J., Stromberg, E., Nowell, D., Forrest, W. (1999). Gray leaf spot: a disease of global importance in maize production. *Plant Disease*, 83(10), 884-895. 



