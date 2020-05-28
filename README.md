# Emotion Recognition

## Introduction

The aim of this report is to cover the design and development of an application for a human emotion recognition project that has been developed as part of the coursework task for the CMP304 Artificial Intelligence module at Abertay University, Dundee.  

It is often taken for granted just how easily a human can identify emotion. Some humans are better than others; they can connect more deeply with other humans on an emotional level and can even feel what another person is feeling simply by metaphorically putting themselves in another’s shoes. Even under a variety of different conditions like lighting, aging, expression, glasses, and hairstyle changes, it is still trivial for a human to decipher the emotion of another as opposed to computers. Humans usually still have no problem with recognising emotions even though some emotions can look similar (e.g. determination and anger). 

If computers were able to accurately recognise any emotion that a human is experiencing, it would help to advance fields in many areas. In fact, there are already many uses for emotion recognition software. 

The current applications of emotion recognition software are mainly used in the field of user-centred design. Three of the ways in which emotion recognition software is currently being used are market research, digital advertising, and one-on-one interviews (Slightcorp Technologies, 2019). They are all used for similar reasons too, mainly to analyse how a user is feeling at a certain time. 

Such software can be used to determine a person’s feeling and comfort level. Practical applications of the software include identification of medical states such as autism. Regarding autism, such software could also be used to help autistic people recognise emotions in other people (Baron-Cohen, et al., 2009). Technology in cars can be enhanced by the software because it can be used to determine the driver’s tiredness and anger levels. Combined with self-driving cars this is useful because the car can take over from the driver in such situations, resulting in safer roads (Kaliouby, 2017). Another application of emotion recognition software lies within the field of software and games testing. Testers would have their emotions read as they are testing the product to give designers an idea of which parts of their product are frustrating and which parts are exciting (Kolakowska, et al., 2013). 

Emotion can be recognised through facial expressions and body language such as hand gestures and posture. Computers can also detect emotion through a human’s voice through speech patterns (Petrushin, 2000). For this project, only facial expressions will be used to recognise emotions.  

The application to be developed should be able to accurately determine the emotion in a picture of a human face. The emotions to be predicted are: joy, anger, sadness, disgust, fear, and surprise. A user to the application will be able to pass in an image of a face and in turn the application will use learned knowledge to determine what emotion is present in the face with a percentage indicating the confidence in whether the answer is correct. 

This report is organised into different sections. The methodology section will cover the steps taken and methodologies used in carrying out the project. The results section will cover test cases and discuss how well the application has performed in accordance to the test cases. 

## Methodology

The OpenCV library (OpenCV Team, 2019) (Python bindings) was used to develop the application. OpenCV is popular for real-time image processing and is very useful because it contains several computer vision algorithms. Three of these algorithms are contained as part of the FaceRecognizer class (OpenCV Team, 2014), these are: Fisherfaces (Martinez, 2011), Eigenfaces (Zhang & Turk, 2008), and Local Binary Patterns Histograms (Pietikainen, 2010).  The FaceRecognizer class encapsulates the functions of the different algorithms and provides an interface consisting of methods like `train()`, `read()`, `save()`, and `predict()`.  

As the name suggests, FaceRecognizer algorithms are designed to recognise patterns in a person’s face in order to recognise the identity of the person, not their emotion. However, it will be interesting to see how well the algorithms perform with recognising emotions. Normally, the FaceRecognizer would take in labels stating the person’s name. Instead, the labels will be names of emotions. The FaceRecognizer class makes it trivial to change the algorithm used, therefore all three of the algorithms will be used to see which one gives the best results. 

### Step 1: Pre-Processing

The MUG facial expression dataset (Aifanti, et al., 2010) will be used for this project. It contains 401 images of 86 different people performing different facial expressions corresponding to an emotion. All the images in the dataset are of the same size and are shot under very similar lighting conditions and background. The images from the dataset have been sorted into folders representing each emotion: anger, disgust, fear, happy, neutral, sadness, and surprise.  

Several processing steps need to be carried out on each of the images in order for the faces to be detected and then subsequently used for training and classification.  

As with facial recognition techniques, the algorithm used for emotion recognition is sensitive to many conditions including light brightness and direction, shadows, face orientation, facial expression, and mood (Baggio, 2012) (Bienvenido B. Abad, 2018). The training will perform exceptionally better when all faces are adjusted to look clearer and more similar to other faces because this will reduce variability in the images. It is important that the images still contain the eyebrows, eyes, and mouth of the person as these are key features in determining an emotion.  

The main goal of the pre-processing stage is to reduce redundancy and irrelevancy in the images. 

The pre-processing steps that were carried out for each image are as follows: 

1. The image was loaded in using OpenCV’s  function. Then, the image was converted to grayscale using OpenCV’s  function to reduce the colour channels that the algorithms have to work with and because the FaceRecognizer class along with most OpenCV methods require that the input images are grayscale. Finally, the image was resized to reduce variability and to improve performance. 

2. OpenCV’s HAAR cascade of classifiers were used to detect a face in the image, namely the default frontal face and eye classifiers (OpenCV Docs, 2018). It is a pretrained model contained in an XML file. The detected face was then cropped out of the image to eliminate information that is not related to recognising emotion in a face (i.e. the background). 

3. Geometrical transformation was carried out on the image. Geometrical transformation involves normalisation of the image to make sure all the faces are aligned which will in turn improve prediction accuracy (Baggio, 2012). The face was rotated to make the two eyes horizontal by making use of a rotational matrix provided by OpenCV. The face was then scaled to ensure that the distance between the two eyes are the same for every image. The eyes were then centred horizontally and positioned at a desired height by using translation. Useless parts of the image were cropped out, such as the background, hair, ears, and chin. This was done using OpenCV’s warpaffine() function. 

4. Histogram equalisation was carried out on the image to improve the contrast and brightness – useful for Eigenfaces and LBPH which are sensitive to light (Bienvenido B. Abad, 2018). This was done using OpenCV’s equalizeHist() function. 
  
5. The image was smoothed using OpenCV’s bilateralFilter() function. This smooths the image but keeps edges sharp. 
  
6. An elliptical mask was extracted from the image. This was done to remove parts of the image that are not useful to emotion recognition, such as the corner regions, including the background and neck. This was implemented by using OpenCV’s ellipse() drawing function. Now the face image is more concise because redundant parts of the image have been removed. 
  
7. Finally, the fully pre-processed image was resized to 70x70 pixels to save storage space and to speed up computation. 
  
### Step 2: Feature Extraction

For humans, the best features for facial extraction would be the eyebrows, eyes, nose, mouth, and cheeks. Combinations of these facial features can be used to express emotions. For example, if a face had wide-open eyes and mouth with raised eyebrows, it could be recognised as surprised, however if the face had lowered eyebrows it could be recognised as angry. The positions of these facial features are used to build a feature vector that consists of the distances and angles between the positions. The number of features should be kept to a minimum in order to reduce classification error and to improve system performance. 

Two of the most commonly used methods to extract facial features are geometric feature-based methods and holistic based methods. Geometric feature-based methods form a feature vector from the outline of the face and position of distinct features including the mouth, eyes, eyebrows, and nose. Holistic based methods use the whole face rather than just the distinct features. (Zahraddeen, et al., 2016) 

Eigenfaces uses the Principle Component Analysis (PCA) algorithm. It forms eigenvectors to create eigenfaces that are added together to make an average face.  Fisherfaces uses Linear Discriminant Analysis (LDA) algorithm. Unlike the other algorithms, this one is not affected by the external conditions like brightness and contrast. This method requires more samples of faces per person (or in this case, emotion) to allow for good extraction of the distinct features. LBPH is different in that it uses a holistic based approach with a sliding window. Instead of using facial features, it finds edges in the images and forms patterns based on this. (OpenCV Team, 2014) 

### Step 3: Learning

Now that the images have been pre-processed (as described in step 2), they are ready to be passed as an input to the FaceRecognizer training function. 

For each emotion in the dataset, all the images were split into an 80/20 ratio where 80% is assigned to training and the remaining 20% are assigned to prediction. This was done for each emotion to ensure that there is a sufficient amount of training and prediction data for each emotion. 

Once the images were passed to the training function and the training was completed, the resulting trained model was saved as a YML file. This is useful in allowing users to simply enter one face image for the application to return a result without having to train the model again because it is very resource intensive and time consuming to train the FaceRecognizer. 

### Step 4: Classification

Now that the system has been trained on a large training set, the system will now try to predict the emotions in the prediction set. The prediction set and labels are passed to the FaceRecognizer predict function. This returns the label that the FaceRecognizer has predicted along with a confidence value (the lower the confidence number, the more confident it is). This will be done 5 times for each algorithm to determine the average accuracy and confidence levels.  

## Results

### Application Interface

The performance was measured by using the timer module in Python. For each significant part of the application, the timer measured how long it took to carry out the task. These times are indicated within the appropriate heading under this results section.  

The application was developed with a command-line interface menu. The following table shows the different parts of the program in use with screenshots. The screenshots have been taken when using the LBPH algorithm. 

### Processing

Out of the 407 images in the MUG dataset, 51 failed to process due to the face or eyes not being detected. The processing could be improved by using facial landmark detection or a better version of the HAAR cascade classifiers that extract the two eyes separately and can extract eyes even when the person is wearing glasses. Nonetheless, despite 51 images failing to process, the dataset was still left with a good number of images to provide for good generalisation. 

Processing 356 images took 59 seconds in total, with each image taking an average of 0.15 seconds to process.  

The usefulness of these pre-processing steps was tested by performing training and prediction on a dataset that has not went through all the pre-processing steps – only converting to grayscale and resizing was performed on each image.  

With the pre-processing steps eliminated and the data looped five times for each algorithm, Fisherfaces returned 57% accuracy, Eigenfaces returned 65% accuracy, and LBPH returned 64% accuracy. The next section will detail the training and prediction results from a fully pre-processed dataset for comparison. 

### Training and Prediction

For the initial stage of testing the prediction accuracy of the application, the dataset was split into 80/20 for each emotion, with 80% of the images going to the training set and the remaining 20% going to the prediction set.  

The training and prediction steps were repeated 5 times in a loop to determine an average accuracy overall and for each emotion. Every image that the predictor got wrong was flagged for further investigation. The results were received by counting every time the predictor got one wrong by comparing the guessed result to the actual result as depicted in the prediction label corresponding to the prediction data. This was done for each of the three algorithms. 

#### LBPH

The LBPH algorithm had an average accuracy of 77%. It was most accurate in determining disgust emotions at 98%, with happy not far behind at 93%. It was most inaccurate at detecting fear emotions at 58%.  

The overall results were very consistent, all of them keeping within the range between 74% and 79%. There are no results where LBPH had 0% accuracy.  

#### Fisherfaces

The Fisherfaces algorithm had an average accuracy of 69%. It was most accurate in determining disgust emotions at 83%, with sadness being the second most accurate at 80%. It was not very effective in recognising neutral faces, scoring 40% accuracy.  

The overall results are inconsistent because loop 2 has a low accuracy of 41% but in loop 4 it has an accuracy of 87%. This can also be seen within the emotion results, loop 1 of neutral has 80% accuracy while loop 4 has 0% accuracy. 

#### Eigenfaces

The Eigenfaces algorithm had an average accuracy of 76%. It was most accurate in determining happy emotions, scoring 87%. It did have trouble determining neutral emotions, with an average accuracy of 48% and a result of 0% in loop 4 but a result of 100% in loop 1. 

#### Comparison

The results comparison chart shows the differences between each of the algorithms: LBPH, Fisherfaces, and Eigenfaces, with LBPH taking the lead. LBPH is also the most reliable, with its results never going below 58%, which is good in comparison to the other algorithms where their lowest values were 40% for Fisherfaces and 48% for Eigenfaces. 

In contrast to the training and prediction carried out with no pre-processing steps done on the images, it is clear to see that the pre-processing steps were very effective in improving the emotion recognition accuracy. This would be because the algorithms are sensitive to certain conditions that the pre-processing steps eliminate. Fisherfaces improved by 12%, going from 57% to 69%. Eigenfaces improved by 11%, going from 65% to 76%, and LBPH improved by 13%, going from 64% to 77%. 

While around 75% is a good accuracy, an accuracy of over 90% would have been more ideal. 

With the LBPH algorithm, training the model took 1.5 seconds on 282 images of faces, while prediction took 3.5 seconds on 68 images of faces. 

## Conclusion

This project, to develop an emotion recognition application, went through the steps of obtaining a dataset – in this case the MUG dataset – before passing it through the preprocessing steps, then performing supervised training, and finally classification to predict emotions in faces that the application has not seen before.  

The training and classification were performed using OpenCV’s FaceRecognizer class, with each of the three algorithms: LBPH, Fisherfaces, and Eigenfaces, being used to discover the most accurate one. 

The pre-processing steps included face and eye detection, cropping and resizing, performing histogram equalisation and geometrical transformation in order to normalise each image to reduce variability because the FaceRecognizer algorithms are sensitive to external conditions such as light. Pre-processing was mostly successful, with 12% of images not being processed correctly (51 out of 407). This did not affect the training or prediction because there were still a large number of images left to provide for good generalisation. It would be improved with better face and eye detection as this is what was going wrong during the processing stages.  

The pre-processing steps were very successful in improving the accuracy for the emotion recognition application. The MUG dataset was passed as input to the algorithms without any pre-processing steps being carried out (except from grayscale and resize as these are required) and the result was that each algorithm performed around 10% less accurately than with the pre-processed dataset. 

LBPH was the most accurate algorithm. Fisherfaces was very inconsistent. Eigenfaces was better than Fisherfaces, but models trained with Eigenfaces also showed inconsistent results, with neutral scoring 0% accuracy in loop 3. LBPH was decided to be the best algorithm to use for this application. The prediction for LBPH is rather accurate, at an average of 75%, but an average accuracy of over 90% would have been more ideal so this still leaves room for improvement. For this reason, this application could not yet be used as a reliable solution in production environments. 

Further testing was carried out on the LBPH algorithm with images of faces that were not within the MUG dataset used to train the model. It performed well, however there were a few faces that could have been described as two emotions. From the results, it is clear that even for a computer it can be a difficult task to recognise the emotion in a face. Emotions can look very similar and usually depend on the context in which the emotion originates.  

Emotion recognition is often highly dependent on the context in which an emotion originates. Just because a person’s face expression shows that they are happy, it does not mean that they actually are. People are known to hide their true emotions and “put on a front”. It can be a difficult task to tell if a person’s smile is genuine if other things are not taken into account like their posture and tone of voice. The accuracy of this emotion recognition application would be greatly improved by combining other methods of emotion recognition such as through voice, body language, and hand gestures. 

Overall, the application was developed successfully and can accurately recognise the emotion in most faces. There are a few improvements that could be made to the application if a second version were to be developed. The application could be developed to be more user-friendly by moving away from a command-line interface in favour of a graphical user interface. The application could be further improved by implementing a camera that recognises and displays emotion in real-time. Furthermore, the application could become more reliable in recognising emotions by integrating extra emotion recognition methods like through posture and speech. 

# References

Aifanti, N., Papachristou, C. & Delopoulos, A., 2010. The MUG Facial Expression Database, Desenzano: s.n.

Baggio, D. L., 2012. Step 2: Face preprocessing. In: Mastering OpenCV. s.l.:Packt Publishing, p. 342.

Baron-Cohen, S., Golan, O. & Ashwin, E., 2009. Can emotion recognition be taught to children with autism spectrum conditions?, Cambridge: Cambridge University.

Bienvenido B. Abad, J., 2018. Proposed Image Pre-Processing Techniques for Face Recognition using OpenCV, Tuguegarao City: St. Paul University of the Philippines.

Kaliouby, R. E., 2017. Driving Your Emotions: How Emotion AI Powers a Safer and
More Personalized Car. [Online]
Available at: https://blog.affectiva.com/driving-your-emotions-how-emotion-aipowers-a-safer-and-more-personalized-car [Accessed 25 March 2019].

Kolakowska, A. et al., 2013. Emotion Recognition and its Application in Software Engineering, Gdansk: IEEE.
Martinez, A., 2011. Fisherfaces. [Online]
Available at: http://www.scholarpedia.org/article/Fisherfaces [Accessed 26 March 2019].

OpenCV Docs, 2018. Face Detection using Haar Cascades. [Online]
Available at: https://docs.opencv.org/3.4.2/d7/d8b/tutorial_py_face_detection.html [Accessed 27 March 2019].

OpenCV Team, 2014. Face Recognition with OpenCV. [Online]
Available at: https://docs.opencv.org/3.0beta/modules/face/doc/facerec/facerec_tutorial.html [Accessed 29 March 2019].
OpenCV Team, 2019. OpenCV. [Online] Available at: https://opencv.org/ [Accessed 25 March 2019].

Petrushin, V. A., 2000. Emotion Recognition in Speech Signal, Northbrook: Center for Strategic Technology Research .
Pietikainen, M., 2010. Local Binary Patterns. [Online]
Available at: http://www.scholarpedia.org/article/Local_Binary_Patterns [Accessed 26 March 2019].

Slightcorp Technologies, 2019. 3 Ways companies are using Emotion Detection technology. [Online]
Available at: https://sightcorp.com/blog/3-ways-companies-are-using-emotiondetection-technology/
[Accessed 25 March 2019].

Zahraddeen, S., Yusuf, A. A., Mohamad, F. S. & Nuhu, A., 2016. Feature Extraction Methods for Face Recognition, Terengganu: Research India Publications.
Zhang, S. & Turk, M., 2008. Eigenfaces. [Online]
Available at: http://www.scholarpedia.org/article/Eigenfaces [Accessed 26 March 2019].
