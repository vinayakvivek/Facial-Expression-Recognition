# FACIAL EXPRESSION RECOGNITION
> Course Project for Foundations of Machine Learning(CS 403)

- **[Project Proposal](http://bighome.iitb.ac.in/index.php/s/jitEohbX01XdD0I)**

### Dependencies
- numpy
- pandas
- matplotlib.pyplot
- `scikit-learn`
- `dlib`
- `opencv`

## Usage
- run `python main.py` to train models.
  - [if you don't wan't to train] for testing models, update lines 17, 18, 19 
  - ie, just to test the MLP classifier, update line 17 to `train_net = False`
- to find the face expression from an image of face, run
  ```python
  python face_expression.py <image_location> <trained_model_location>
  ```
    
