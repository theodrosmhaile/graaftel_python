# graaftel_python
This is a python translation of Graaftel algorithm by N.A. Taatgen. 
See here for the original Swift implementation: https://github.com/ntaatgen/ELO

## To train new data:

 1. create a new Model object by calling Model(). Perhaps called 'm'.

 2. Initiate model with new data by calling the method m.init_model(data). Passing a path to a CSV file to data is required here. Other parameters if different from the default should be set at this time.
 
 3. Finally run the method m.get_ratings()

## To load a model and optinally train further, perhaps with new data of the same students with new questions or the same questions with new students. 

 1. create a new Model object by calling Model(). perhaps called 'm'.
 
 2. load previously trained model by calling the method m.load_model(model, data). Parameters 'model' and 'data' are required. Other parameters will be loaded from the model file, but can be optionally set here. Note that these should be set at this time if new parameters are needed. 
 
 3. Run method m.get_ratings() if more training is needed. 

## To get ratings for one student (assumes a model 'm' has been trained and/or loaded):
1. If student already exists in the model, simply run m.update_student_rating(name, item, score), by providing the name of the  
  student, the question item, and the score they obtained. A new record will be added to the list of scores and the student's skills will be updated. 

2. If it is a new student with no previous scores, first run m.new_student(name). Then run step 1 above. 


## Other Functions:
 
### Model object 'm', that has been trained, can be passed to these other functions from GraafTel: 
1. make_nodes(model, graph=False)
 Pass the model object m to generate nodes and edges, optionally set graph to True to plot a basic graph. 
    
2. calculate_error(model, student=None)
This computes the error of the predicted score (actual score - predicted score) for all the questions in the model for all students. If student is set to a student ID, the predicted error for that student only is returned. 

3. Get_recommendations(model, student=None)
This returns the Id's and predicted scores of all the questions that fall with-in the range of "acceptable difficulty" for the student. 
If student is set to an Id, only that student's recommended questions are returned. 
