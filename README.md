# Cohere Health Learning Assignment
## Author: Vinayak (Vin) Kannan

Thank you for this opportunity! Please see below for instructions on how to run the 'database' generator and the 'Bonus Task' diagnostic tool

## How to run 'database' generator for sampleclinicalnotes
1. Obtain an OpenAI key (this is used to facilitate raw text cleanup and the LLM model in the ensemble method detailed in the related PDF report). This key needs to have access privelidges to the gpt-4 model. If you need a key, please text me at 734-263-4313 or email me at vk2364@columbia.edu. Alternatively, you can skip dataset generation by proceeding to the 'Testing / Validation output' cell.
2. Go to 'Project/OutputDatasetCreator/dataset_generator_drive.ipynb'
3. Run all cells. Note that most helper code is in a seperate 'Helper' directory; the notebook is to enable easy code setup and to combine both the dataset_generator_driver and the test_driver in one location
4. If you wish to skip to the output, please do so by going to the 'Diagnosis and Symptom Explorer' cell in the notebook and continuing. These cells enable you to search for Primary Medical Diagnoses and view their related Common Underlying Factors. In addition, you will see output from the ensemble method used in this repository; this can help give guidance to the end-user on the confidence of the methodology and potentially flag when the end-user should consider manually reviewing the data
