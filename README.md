### plant-seedlings
This repo is for a image classification analysis using Plant Seedlings data provided by [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification) and _scikit-learn_ ensemble classifiers.

#### Results
See [_Plant Seedling Classification - Results_](Plant%20Seedling%20Classification%20-%20Results.ipynb) for summary of results.

To download the image data used in this analysis, install the Kaggle python module and use command

`kaggle competitions download -c plant-seedlings-classification`

To configure paths and other settings, adjust `settings.py`

To create the converted images, install [GIMP](https://www.gimp.org) and the [BIMP](https://alessandrofrancesconi.it/projects/bimp/) plugin, then use the provided BIMP scripts in the `./bimp_scripts` folder of this repository.

Please ensure that all required modules are installed. Use the inclded _requirements.txt_ file

`pip install -r requirements.txt`

#### Future improvements
Instead of using an external application, I want to try using built-in OpenCV function to apply image processing programatically. I also want to try to use a mask to isolate the plant part of the image, something I attempted with Gimp but could not easily automate.

Further, scikit-learn ensemble methods may not be best suited for this task. I will try keras next as that library appears to yield better results with this type of data.
