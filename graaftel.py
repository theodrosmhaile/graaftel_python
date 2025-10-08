"""
Written by Niels Taatgen 2024
 Python implementation by TMH

To train new data:

 1. create a new Model object by calling Model(). Perhaps called 'm'.
 2. Initiate model with new data by calling the method m.init_model(data). Passing a path to a CSV file to data is required here. 
Other parameters if different from the default should be set at this time.
 3. Finally run the method m.get_ratings()

To load a model and optinally train further, perhaps with new data of the same students with new questions or the same questions with new students. 
 1. create a new Model object by calling Model(). perhaps called 'm'.
 2. load previously trained model by calling the method m.load_model(model, data). Parameters 'model' and 'data' are required.
   Other parameters will be loaded from the model file, but can be optionally set here. Note that these should be set at this time if new parameters are needed. 
 3. Run method m.get_ratings() if more training is needed. 

To get ratings for one student (assumes a model 'm' has been trained or loaded):
1. If student already exists in the model, simply run m.update_student_rating(name, item, score), by providing the name of the  
  student, the question item, and the score they obtained. A new record will be added to the list of scores and the student's skills will be updated. 
2. If it is a new student with no previous scores, first run m.new_student(name). Then run step 1 above. 


 Other Functions:
 
 Model object 'm', that has been trained, can be passed to these other functions from GraafTel: 
1. make_nodes(model, graph=False)
 Pass the model object m to generate nodes and edges, optionally set graph to True to plot a basic graph. 
    
2. calculate_error(model, student=None)
This computes the error of the predicted score (actual score - predicted score) for all the questions in the model for all students. If student is set to a student ID, 
the predicted error for that student only is returned. 
3. Get_recommendations(model, student=None)
This returns the Id's and predicted scores of all the questions that fall with-in the range of "acceptable difficulty" for the student. 
If student is set to an Id, only that student's recommended questions are returned. 
"""


import numpy as np
import pandas as pd
import math
import random
import networkx as nx
import json


## Classes

### student Class
class Student:
    def __init__(self, name, nSkills, skills=None, m=None, v=None, t=None):
        self.skills = skills if skills is not None else[random.uniform(0.15, 0.18) for i in range(nSkills)]
        self.name = name
        self.m = m if m is not None else [0]*nSkills
        self.v = v if v is not None else [0]*nSkills
        self.t = t if t is not None else 1
        self.error = []

    def __repr__(self):
        return " %s :%s" % (self.name, [round(self.skills[i],2) for i in range(len(self.skills))])
   
    
### Question item Class
class Item:
    def __init__(self, name, nSkills, skills=None, m=None, v=None, t=None):
        self.skills = skills if skills is not None else [random.uniform(0.48, 0.52) for i in range(nSkills)]
        self.name = name
        self.experiences = 0
        self.m = m if m is not None else [0]*nSkills
        self.v = v if v is not None else [0]*nSkills
        self.t = t if t is not None else 1
    
    def __repr__(self):
        return "%s %s" % (self.name, [round(self.skills[i],2) for i in range(len(self.skills))])
        
### Question item Class
class Score: 
    def __init__(self, student, item, score):
        self.student = student
        self.item = item
        self.score = score
    
    def __repr__(self):
        return "Score student:%s item:%s score:%s" % (self.student, self.item, self.score)

class node:
    def __init__(self, binary_vector, str_id):
        
        self.skills = binary_vector 
        self.name = str_id
        self.connects_to = []
        self.height = sum(self.skills) #this is used to limit connections to the layer immediately above
        self.questions = []

class Model:
    def __init__(self):
#         Scores, Students, Items, dat, nSkills
        self.scores = {}#Scores
        self.students = {}#Students
        self.items = {}#items
        self.data = []#dat
        self.nskills = None #nSkills
        self.nodes = []
        
    def save(self, file_name):
        """
        Save all the necessary objects as JSON to file. 
        This might also help compatibility with web apps etc hopefully
        """
        temp = {"alpha": self.alpha,
                 "nEpochs": self.nEpochs,
                 "nSkills":self.nskills,
                 "items": { str(i):{ "skills":self.items[i].skills, 
                                     "name" : str(i), 
                                     "m" : self.items[i].m, 
                                     "v": self.items[i].v, 
                                     "t": self.items[i].t
                                   } for i in self.items
                          }, 
                 "nodes": { str(i): {'height': self.nodes[i].height,'connects_to': self.nodes[i].connects_to} for i in self.nodes},
                 "students": {str(s): {'skills': self.students[s].skills, 
                                       "m" : self.students[s].m,
                                       "v" : self.students[s].v, 
                                       "t" : self.students[s].t
                                      } for s in self.students
                             }
                }
        
        
        with open(file_name + '.JSON', 'w') as fp:
            json.dump(temp, fp)
            
    def load_model(self, model, data,studentMode=False, nSkills=4, alpha = None, nEpochs=None):
        """
        Revert the JSON to dictionaries of objects to continue running in models. 
        Have to be careful not to lose data. 
        """
        with open(model) as fp:
            M = json.load(fp)
       
        
       # M = m_import#['model']
        self.data = pd.read_csv(data, header=None)
        
        self.scores = {i:Score(student = str(self.data[0][i]), item = str(self.data[1][i]), score= self.data[2][i]) 
                       for i in range(len(self.data))}
        
        self.students = {s:Student(name = s,
                                  nSkills = M['nSkills'], 
                                  skills = M['students'][s]['skills'],
                                   m = M['students'][s]['m'], 
                                   v = M['students'][s]['v'], 
                                   t = M['students'][s]['t']
                                   
                                  ) for s in M['students'].keys()}
        
        self.items = {q:Item(name = q,
                             nSkills= M['nSkills'], 
                             skills = M['items'][q]['skills'],
                             m = M['items'][q]['m'],
                             v = M['items'][q]['v'],
                             t =  M['items'][q]['t']
                             
                             ) for q in M['items'].keys()
                     }
        
        self.nskills = nSkills if nSkills is not None else M['nSkills']
        #self.nodes = M['nodes']
        self.nEpochs = nEpochs if nEpochs is not None else M['nEpochs']
        self.alpha = alpha if alpha is not None else  M['alpha']
        self.studentMode  = studentMode 

    def init_model(self, data, studentMode=False, nSkills=4, alpha = 0.0005, nEpochs=1000):
       #
    ## Import data
        self.data = pd.read_csv(data, header=None)
    
    ### first create a dictionary of score objects from the imported data called Score
        self.scores = {i:Score(student = str(self.data[0][i]), item = str(self.data[1][i]), score= self.data[2][i]) for i in range(len(self.data))}
    
    ### then create a dictionary of Students using the Student object
        self.students = {str(s):Student(name = s, nSkills=nSkills) for s in self.data[0].unique()}
    
    ### then create a dictionary of Items using the Item object
        self.items = {str(q):Item(name = q, nSkills=nSkills) for q in self.data[1].unique()}
    
        self.nskills = nSkills
        self.alpha = alpha
        self.nEpochs = nEpochs
        self.studentMode = studentMode
    
    def get_ratings(self):
        
    ### Specify a number of nEpochs to run skill rating -  make sure it starts at random points in each epoch
    ### OneItemAdam iterates over all available scores but takes only one item at a time and updates the skllils in the objects
        
        for e in range(self.nEpochs):
            rand_indexes = random.sample([i for i in range(len(self.scores))], len(self.scores)) ## for random starts
            for i in rand_indexes:
    
                oneItemAdam(score = self.scores[i], Students = self.students, Items = self.items, nSkills = self.nskills, 
                           studentMode = self.studentMode, alpha= self.alpha,
                            beta1 = 0.9, beta2 = 0.999, epsilon= 1e-8, alphaHebb = 1.0)


    def new_student(self, name):
        
        self.students[name] = Student(name = name, nSkills = self.nskills)
        

    def update_student_rating(self, name, item, score):
        
        oneItemAdam(score = Score(student=name,item=item, score=score),
               Students=self.students,
               Items=self.items, 
               studentMode=self.studentMode,
               nSkills=self.nskills,
               alpha=self.alpha)

        self.scores[len(self.scores)] = Score(student=name, item=item, score=score)
        
       
       
    
    
## Calculate the predicted score base on a single skill, given a student score and an item score.
   # - Parameters:
    #   - studentDifficulty: The student score, between 0 and 1.
    #   - itemDifficulty: The item score, between 0 and 1.
    # - Returns: The expected score for one skill.

        
def calcProb(studentDifficulty, itemDifficulty):
    return 1 - itemDifficulty + itemDifficulty * studentDifficulty
      
## expected score function
def expectedScore(student, item, nSkills, leaveOut=None):
    p = 1

    for i in range(nSkills):
        if leaveOut == None or leaveOut != i: 
            skillP = calcProb(studentDifficulty= student.skills[i], itemDifficulty= item.skills[i])
            p = p * skillP
    
    return p


#####Add to two numbers, but keep them between a lowerbound and an upperbound
# - Parameters:
#   - num1: The first number
#   - num2: The second number
#   - lwb: The lowerbound, 0 by default
#   - upb: The upperbound, 1 by default
# - Returns: The bounded sum
def boundedAdd(num1, num2, lwb = 0.0, upb = 1.0):
    
    s = num1 + num2
    if s < lwb: 
        return lwb
    
    elif s > upb:
        return upb 
    
    else: return s 
    


def oneItemAdam(score, Students, Items,studentMode, alpha= 0.001, nSkills = 4, beta1 = 0.9, beta2 = 0.999, epsilon= 1e-8, alphaHebb = 1.0):
    """
Update the model based on a single datapoint using Adam optimization
 - Parameters:
   - score: The datapoint used for the update
   - alpha: The alpha parameter for Adam, 0.001 by default
   - beta1: The beta1 parameter for Adam, 0.9 by default
   - beta2: The beta2 parameter for Adam, 0.99 by default
   - epsilon: The epsilon parameter, 1e-8 by default
   - alphaHebb: Learning multiplier (with alpha) to control the Hebbian learning.
  """
    s = Students[score.student]
    it = Items[score.item]
    error = score.score - expectedScore(student= s, item= it,nSkills = nSkills)
    expectedWithoutSkill = []
    
    for i in range(nSkills): 
        expectedWithoutSkill.append(expectedScore(student = s, item = it, nSkills= nSkills,leaveOut = i))
        
    for i in range(nSkills): 
            
        itGradient = -2 * error * expectedWithoutSkill[i] * (s.skills[i] - 1)
        it.m[i] = beta1 * it.m[i] + (1 - beta1) * itGradient
        it.v[i] = beta2 * it.v[i] + (1 - beta2) * pow(itGradient, 2) 

        mhatI = it.m[i] / (1 - pow(beta1, it.t))
        vhatI = it.v[i] / (1 - pow(beta2, it.t))

        sGradient = -2 * error * expectedWithoutSkill[i] * it.skills[i]
        s.m[i] = beta1 * s.m[i] + (1 - beta1) * sGradient
        s.v[i] = beta2 * s.v[i] + (1 - beta2) * pow(sGradient, 2)


        mhatS = s.m[i] / (1 - pow(beta1, s.t))
        vhatS = s.v[i] / (1 - pow(beta2, s.t))

        if not studentMode: 
            it.skills[i] = boundedAdd(it.skills[i], -alpha * mhatI / (math.sqrt(vhatI) + epsilon))
            s.skills[i] = boundedAdd(s.skills[i],  -alpha * mhatS / (math.sqrt(vhatS) + epsilon))
        else: 
            #print(s.skills)
            s.skills[i] = boundedAdd(s.skills[i], -alpha * sGradient)
        

        
    it.t += 1
    s.t += 1

    it.experiences += 1 # redundant





def calculateError(model, student=None): ## This function computes the final error after model runs
    """
    Calculate the average error per datapoint, either of the whole dataset, or the last loaded students.
     - Returns: The average error
    """
    errors = []
    count = 0
    ## instances takes all scores, and therefore all students and questions unless a student is specified
    instances = model.score if student is None else [i for i in model.scores if model.scores[i].student== student]
   
    for s in instances:
        try:
          
            errors.append([model.students[model.scores[s].student].name, 
                           model.items[model.scores[s].item].name,
                           
            math.sqrt(pow(model.scores[s].score - expectedScore(
                    student = model.students[model.scores[s].student],
                    item = model.items[model.scores[s].item], 
                    nSkills = model.nskills),2
                                      )
                                  )
                          ]
                         )
            count += 1
            
        except:
            print('item:',model.scores[s].item,' not in Train data for student ',model.scores[s].student )
            pass
    #print('error based on ', str(count), ' items')              
    return errors


def make_nodes(model, graph=False, threshold=0.5):
    ## first convert the skill vectors to binary using the provided data and threshold. Also make str names
    items_binary = [[[q],
                 [int(e >= threshold) for e in model.items[q].skills],
                 ''.join(map(str, [int(e >= threshold) for e in model.items[q].skills]))
                ] for q in model.items.keys()
            ]
    
    ## Generate a dict of unique nodes
    nodes = {items_binary[i][2]: node(binary_vector=items_binary[i][1], str_id=items_binary[i][2]) 
                    for i in range(len(items_binary))}
   
      ### update nodes with the connections      
    
### iterate through the "connect from" nodes
    for n1 in nodes:
        
        ### iterate through the "connect to" nodes
        for n2 in nodes:
            
            ### Make connections only to the nodes that are 1 step higher in the heirarchy 
        
            if nodes[n2].height - nodes[n1].height == 1:
                if sum(nodes[n1].skills) == 0: ## first node always connects to all skills so this is a special case
                    nodes[n1].connects_to.append(n2)
                    
                else: 
                    ## first create a mask for the 'to' node (n2) using the 'from' node(n1)
                    ## make a link if all the masked slots in n2 are equal to 1
                    if all([n2[i]=='1' for i in range(len(n1)) if n1[i]=='1']):
                        nodes[n1].connects_to.append(n2)

            ### Here, check if all nodes at n+1 level exist, if True, passs, if False check nodes at n+2
    
    ###Round 2: Check missing links that go beyound 1 level and add them
    ### Not likely that this will run often but needed just in case. 
    for i in nodes:
    ## for this node's current location in the heirarchy, there should be (number of skills - height) number of connections 
        expected_n_cons = model.nskills - nodes[i].height
    
    #If  there aren't expected number of nodes to connect to, move forward
        if  expected_n_cons != len(nodes[i].connects_to):
        
        ## exclude node connections we already know about
            ## step 1: get all nodes we know about for 'this' node
            temp = [nodes[n].connects_to for n in nodes[i].connects_to]
            exclude_list = set(sum(temp, [])) ## this just squeezes the list
            
            ## step2: make list of potential connections that go 2 steps higher in the heirarchy 
            ##        we already know everything about nodes 1 step higher in the hierarchy ...
            ##        this is a psuedo complete list because not all nodes might exist at this level. 
            ##        We don't care if they don't exist. 
            pseudo_complete_list = set([n for n in nodes if nodes[i].height + 2 == nodes[n].height])
           # print('current node: ', i)
            #print('pcl: ', pseudo_complete_list)
       
            ## step3: find the difference between the two lists, move forward if not empty list.
            ##        If not empty, maybe there is a potential node that does not have an intermediary. 
            check_list = [n for n in pseudo_complete_list.difference(exclude_list)]
           # print(check_list)
            if check_list != []:
           
            ## step4: check if there is a skill overlap between the node we started with and the potential node from check_list
                for t in check_list:
                    if  all([t[e]=='1' for e in range(len(i)) if i[e]=='1']): #e for vector element
                        print('updated node ',i, ' with ', t) 
                        nodes[i].connects_to.append(t)

    model.nodes = nodes
    
    if graph:
        Graph = nx.DiGraph()
        Graph.add_edges_from([[a,b] for a in nodes for b in nodes[a].connects_to])
        nx.draw_networkx(Graph, pos = nx.bfs_layout(Graph, ''.join(['0' for i in range(model.nskills)])), 
                 arrows=True, 
                 node_size = 1200, 
                node_color='#7fcdbb')

def make_js_nodes(nodes):
    
    js_out = {'edges':
     [{'data':{'source':n1, 'target':n2}} for n1 in nodes for n2 in nodes[n1].connects_to]
    }
    return js_out

def get_recommendations(model, student=None):

    def apply_criterion(val):
        test =  val < 0.95 #val > 0.65 and .75
        return test
    #### harder or easier also 
    ## how to step 
    students = [student] if student is not None else model.students
    rec_items = {}
    items = model.items
    for s in students:
        es = []
        for i in items:

            es_temp = expectedScore(student = model.students[s], item = items[i], nSkills=model.nskills)

            if apply_criterion(es_temp):
                es.append([es_temp,i ])

        es.sort()
        rec_items[model.students[s].name] = es
    
    return rec_items


