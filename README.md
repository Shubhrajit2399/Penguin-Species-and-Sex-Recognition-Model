# Penguin-Species-and-Sex-Recognition-Model
This is a Machine Learning Classification Model where we have identified the Species and Sex of the Penguins with the help of some constraints from the given dataframe.

<img width="418" height="246" alt="Screenshot 2025-09-13 at 1 06 59 PM" src="https://github.com/user-attachments/assets/4bfbb5ee-1935-45df-bed6-1690ed53eacb" />

1. #Step-1: We have analyzed the data and trimmed and cleaned it in (penguine_data_analysis.ipynb)
   
    Identified the data from the file with Sex=NA and kept them for testing. For the rows where other attributes than Sex was also NULL dropped those.

2.  #Step-2: With the cleaned train data created histogram to analyze it.
   <img width="835" height="834" alt="download" src="https://github.com/user-attachments/assets/6c035b2c-4dfc-442c-ab45-d71f78f2d5eb" />

3. #Step-3: Trained different classification models with the train dataset and analyzed the accuracy of every model. (penguine_model_selection.py)
   
   From the model analyzation found the RandomForestClassifier Model is the best choice with higest accuracy (mean) and lowest STD.
   
   <img width="388" height="132" alt="Screenshot 2025-09-13 at 1 21 51 PM" src="https://github.com/user-attachments/assets/370c168d-c3d5-4b42-9749-7989efe363f0" />

4. #Step-4: Created the final prediction model with RandomForestClassifier. (penguine_reconization_final.py)
   
   At first to validate the model is working fine or not, splitted the train_data into pre-train and pre-test data and cross validated the pre-test data with output data.
   Then used the whole train dataset to train the model and used the previously taken out data with Sex=NA to predict the Species and Sex of Penguins.

--End Of File--
