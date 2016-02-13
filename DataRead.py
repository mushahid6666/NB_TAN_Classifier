from __future__ import division
import re
import matplotlib
import random
import math
from scipy.io import arff
import weka.core.converters as converters


randlist = list()
nodes_data = list()
edges_data = list()
parents_data = dict()
Attr_data=[]
TrainDataSet=[]
attributeCounter=0
ComputerWieghts={}
Mutualinfo={}
Cond_Mutual_Info={}
CondProbTable={}
class Attribute():
    name=''
    values=list()
    index=0
    def __init__(self,str,values):
        global attributeCounter
        self.name=str
        self.values=values
        self.index=attributeCounter
        attributeCounter+=1
        self.values_count = {}
        for value in values:
            self.values_count[value]=[0,0]



def InputPares(filename):
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    global CondProbTable

    finput= open(filename,'r')
    # lines = [line.rstrip('\n') for line in finput]
    count = 1;
    for line in finput:
        line=line.rstrip('\n')
        if line.startswith('@attribute'):
            templist =  line.split(' ')
            values = re.findall(r'\{([^]]*)\}',line)
            newlist=values[0].split(",")
            newlist = [each.strip() for each in newlist]
            newlist = [each.strip("'") for each in newlist]
            # templist[1]=templist[1].replace('\\','').replace("'",'')
            templist[1]=templist[1].strip()
            newatr = Attribute(templist[1],newlist)
            Attr_data.append(newatr)
        if line.startswith('@attribute') or line.startswith('@relation') or line.startswith('%') or line.startswith('@data'):
            continue
        else:
            # if count in randlist:
            line=line.strip()
            line= line.split(',')
            line = [each.strip() for each in line]
            line = [each.strip("'") for each in line]
            TrainDataSet.append(line)
            if line[-1]==Attr_data[-1].values[0]:
                for i in range(len(line)):
                    Attr_data[i].values_count[line[i]][0]+=1
            else:
                for i in range(len(line)):
                    Attr_data[i].values_count[line[i]][1]+=1
            # count+=1

def NBClassifier(TestFile):
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    global CondProbTable
    y1=Attr_data[-1].values[0]
    y2=Attr_data[-1].values[1]
    y1.strip("'")
    y2.strip("'")
    testinput= open(TestFile,'r')
    correct_classified=0
    C_X=Attr_data[-1].values_count[Attr_data[-1].values[0]][0]
    C_Y=Attr_data[-1].values_count[Attr_data[-1].values[1]][1]
    total=C_X+C_Y
    PX=(C_X+1)/(total+2)
    PY=(C_Y+1)/(total+2)
    count=1
    for line in testinput:
        if line.startswith('@attribute') or line.startswith('@relation') or line.startswith('%') or line.startswith('@data') or line.startswith('@attribute'):
            continue
        else:
            count+=1
            line=line.strip()
            line= line.split(',')
            line = [each.strip() for each in line]
            line = [each.strip("'") for each in line]

            Px_y=1.0
            Px_y_dash=1.0
            for i in range(len(line)-1):
                CT_Y=Attr_data[i].values_count[line[i]][0]
                Px_y*=(CT_Y+1)/+(C_X + len(Attr_data[i].values))
            result1=Px_y*PX
            for i in range(len(line)-1):
                CT_Y=Attr_data[i].values_count[line[i]][1]
                Px_y_dash*=(CT_Y+1)/(C_Y + len(Attr_data[i].values))
            result2=Px_y_dash*PY
            final_result1=float(result1)/float(result1+result2)
            final_result2=float(result2)/float(result1+result2)
            if final_result1>final_result2:
                if line[-1]==Attr_data[-1].values[0]:
                    correct_classified+=1
                temp = line[-1]
                temp = temp.strip("'")
                print y1,temp,"%.12f"%final_result1
            else:
                if line[-1]==Attr_data[-1].values[1]:
                    correct_classified+=1
                temp = line[-1]
                temp = temp.strip("'")
                print y2,line[-1],"%.12f"%final_result2

    print(correct_classified,count)

def CalculateConditionalMutualInfo():
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    global CondProbTable
    for Att_I in Attr_data:
        if Att_I.index==Attr_data[-1].index:
            continue
        Mutualinfo[Att_I.index]={}
        for Att_J in Attr_data:
            if Att_J.index==Attr_data[-1].index:
                continue
            if Att_I.index==Att_J.index:
                continue
            Mutualinfo[Att_I.index][Att_J.index]={}
            for value_I in Att_I.values:
                Mutualinfo[Att_I.index][Att_J.index][value_I]={}
                for value_J in Att_J.values:
                    Mutualinfo[Att_I.index][Att_J.index][value_I][value_J]=[0,0]
                    for row in TrainDataSet:
                        if row[Att_I.index]==value_I and row[Att_J.index]==value_J and row[-1]==Attr_data[-1].values[0]:
                            Mutualinfo[Att_I.index][Att_J.index][value_I][value_J][0]+=1
                        if row[Att_I.index]==value_I and row[Att_J.index]==value_J and row[-1]==Attr_data[-1].values[1]:
                            Mutualinfo[Att_I.index][Att_J.index][value_I][value_J][1]+=1
    pass

def get_C_xi_xj_given_y(Att_I,value_I,Att_J,value_J,y):
    global Mutualinfo

    return Mutualinfo[Att_I.index][Att_J.index][value_I][value_J][y]

def get_C_xi_xj(Att_I,value_I,Att_J,value_J):
    global Mutualinfo
    return Mutualinfo[Att_I.index][Att_J.index][value_I][value_J][0]+Mutualinfo[Att_I.index][Att_J.index][value_I][value_J][1]

def Prob(Numerator,Denomenator):
    return float(Numerator)/Denomenator

def get_Prob_xi_y(Att_I,value_I,y):
    global Attr_data
    no_of_y1=Attr_data[-1].values_count[Attr_data[-1].values[0]][0]
    no_of_y2=Attr_data[-1].values_count[Attr_data[-1].values[1]][1]
    if y==0:
        CT_I_Values=len(Att_I.values)
        CT_I_given_Y=Att_I.values_count[value_I][0]
        Prob_xi_given_y1=Prob((CT_I_given_Y+1),(no_of_y1+CT_I_Values))
    else:
        CT_I_Values=len(Att_I.values)
        CT_I_given_Y=Att_I.values_count[value_I][1]
        Prob_xi_given_y1=Prob((CT_I_given_Y+1),(no_of_y2+CT_I_Values))
    return Prob_xi_given_y1

def CalculateWeights():
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    CalculateConditionalMutualInfo()
    no_of_y1=Attr_data[-1].values_count[Attr_data[-1].values[0]][0]
    no_of_y2=Attr_data[-1].values_count[Attr_data[-1].values[1]][1]
    Cond_mutual_info_xi_xj = 0
    for Att_I in Attr_data:
        if Att_I.index == Attr_data[-1].index:
            continue
        for Att_J in Attr_data:
            if Att_J.index==Attr_data[-1].index or Att_I.index==Att_J.index:
                continue
            for value_I in Att_I.values:
                for value_J in Att_J.values:
                    C_xi_xj_given_y1=get_C_xi_xj_given_y(Att_I,value_I,Att_J,value_J,0)
                    Prob_xi_xj_given_y1=Prob(C_xi_xj_given_y1+1,(no_of_y1+(len(Att_I.values)*len(Att_J.values))))
                    Prob_xi_xj_y1=Prob(C_xi_xj_given_y1+1,(no_of_y1+no_of_y2+(len(Att_I.values)*len(Att_J.values)*2)))


                    C_xi_xj_given_y2=get_C_xi_xj_given_y(Att_I,value_I,Att_J,value_J,1)
                    Prob_xi_xj_given_y2=Prob(C_xi_xj_given_y2+1,(no_of_y2+(len(Att_I.values)*len(Att_J.values))))
                    Prob_xi_xj_y2=Prob(C_xi_xj_given_y2+1,(no_of_y1+no_of_y2+(len(Att_I.values)*len(Att_J.values)*2)))


                    Prob_xi_given_y1=get_Prob_xi_y(Att_I,value_I,0)
                    Prob_xi_given_y2=get_Prob_xi_y(Att_I,value_I,1)
                    Prob_xj_given_y1=get_Prob_xi_y(Att_J,value_J,0)
                    Prob_xj_given_y2=get_Prob_xi_y(Att_J,value_J,1)

                    Cond_mutual_info_xi_xj_y1=Prob_xi_xj_y1*math.log(float(Prob_xi_xj_given_y1)/(Prob_xi_given_y1*Prob_xj_given_y1),2)
                    Cond_mutual_info_xi_xj_y2=Prob_xi_xj_y2*math.log(float(Prob_xi_xj_given_y2)/(Prob_xi_given_y2*Prob_xj_given_y2),2)

                    Cond_mutual_info_xi_xj+=Cond_mutual_info_xi_xj_y1+Cond_mutual_info_xi_xj_y2
            key=Att_I.name+","+Att_J.name
            Cond_Mutual_Info[key]=Cond_mutual_info_xi_xj
            Cond_mutual_info_xi_xj=0
    # for Att_I in Attributelist:
    #     print Att_I.name,
    #     for Att_J in Attributelist:
    #         if Att_I.index == Attributelist[-1].index or Att_J.index==Attributelist[-1].index or Att_I.index==Att_J.index:
    #              continue
    #         key=Att_I.name+","+Att_J.name
    #         print Cond_Mutual_Info[key],
    #     print
def cmp_items(a, b):
    if a[0]> b[0]:
        return -1
    elif a[0] == b[0]:
        if a[1].index < b[1].index:
            return -1
        elif a[1].index == b[1].index:
            if a[2].index < b[2].index:
                return -1
            else:
                return 1
        else:
            return 1
    else:
        return 1

def getMaxWtEdge(nodes_data):
    global randlist
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    weights=list()
    for node in nodes_data:
        for Attr in Attr_data:
            if Attr.index==node.index or Attr.index==Attr_data[-1].index or node.index==Attr_data[-1].index:
                continue
            if Attr not in nodes_data:
                key=node.name+","+Attr.name
                weights.append([Cond_Mutual_Info[key],node,Attr])
    weights.sort(cmp=cmp_items)
    if len(weights) >0:
        return weights[0]

def PrimsAlgo():
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    nodes_data.append(Attr_data[0])
    while(len(nodes_data)<len(Attr_data)-1):
        maxEdge = getMaxWtEdge(nodes_data)
        nodes_data.append(maxEdge[2])
        edges_data.append(maxEdge)
        parents_data[maxEdge[2].name]=maxEdge[1]
    for Attribute in Attr_data:
        if Attribute.index==Attr_data[-1].index:
            continue
        if parents_data.has_key(Attribute.name):
            temp = Attribute.name
            temp.strip()
            temp.strip("'")
            print temp
            print temp+" "+ parents_data[Attribute.name].name+" class"
        else:
            print Attribute.name+" class"

def ConditionalProbilityTable():
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    y1=Attr_data[-1].values[0]
    y2=Attr_data[-1].values[1]
    for attr_I in Attr_data:
        for value_I in attr_I.values:
            if parents_data.has_key(attr_I.name):
                parent_I=parents_data[attr_I.name]
                for value_J in parent_I.values:
                    C_xI_given_parentxJ_y1=get_C_xi_xj_given_y(attr_I,value_I,parent_I,value_J,0)
                    C_xI_given_parentxJ_y2=get_C_xi_xj_given_y(attr_I,value_I,parent_I,value_J,1)
                    C_parentxJ_y1=parent_I.values_count[value_J][0]
                    C_parentxJ_y2=parent_I.values_count[value_J][1]
                    Prob_xI_given_parentxJ_y1=Prob(C_xI_given_parentxJ_y1+1,C_parentxJ_y1+len(attr_I.values))
                    Prob_xI_given_parentxJ_y2=Prob(C_xI_given_parentxJ_y2+1,C_parentxJ_y2+len(attr_I.values))
                    key1=str(attr_I.index)+","+value_I+","+str(parent_I.index)+","+value_J+","+y1
                    CondProbTable[key1]=Prob_xI_given_parentxJ_y1
                    key2=str(attr_I.index)+","+value_I+","+str(parent_I.index)+","+value_J+","+y2
                    CondProbTable[key2]=Prob_xI_given_parentxJ_y2




def TANClassifier(TestFile):
    global randlist
    global nodes_data
    global edges_data
    global parents_data
    global Attr_data
    global TrainDataSet
    global attributeCounter
    global ComputerWieghts
    global Mutualinfo
    global Cond_Mutual_Info
    CalculateWeights()
    PrimsAlgo()
    ConditionalProbilityTable()
    testinput= open(TestFile,'r')
    correct_classified=0
    C_X=Attr_data[-1].values_count[Attr_data[-1].values[0]][0]
    C_Y=Attr_data[-1].values_count[Attr_data[-1].values[1]][1]
    total=C_X+C_Y
    PX=(C_X+1)/(total+2)
    PY=(C_Y+1)/(total+2)
    y1=Attr_data[-1].values[0]
    y2=Attr_data[-1].values[1]
    count=1
    for line in testinput:
        if line.startswith('@attribute') or line.startswith('@relation') or line.startswith('%') or line.startswith('@data') or line.startswith('@attribute'):
            continue
        else:
            count+=1
            line=line.strip()
            line= line.split(',')
            line = [each.strip() for each in line]
            line = [each.strip("'") for each in line]
            Px_y=1.0
            Px_y_dash=1.0
            for i in range(len(line)-1):
                if parents_data.has_key(Attr_data[i].name):
                    parent_J=parents_data[Attr_data[i].name]
                    value_J=line[parent_J.index]
                    key1=str(Attr_data[i].index)+","+line[i]+","+str(parent_J.index)+","+value_J+","+y1
                    Px_y*=CondProbTable[key1]
                else:
                    CT_Y=Attr_data[i].values_count[line[i]][0]
                    Px_y*=(CT_Y+1)/(C_X + len(Attr_data[i].values))
            result1=Px_y*PX

            for i in range(len(line)-1):
                if parents_data.has_key(Attr_data[i].name):
                    parent_J=parents_data[Attr_data[i].name]
                    value_J=line[parent_J.index]
                    key1=str(Attr_data[i].index)+","+line[i]+","+str(parent_J.index)+","+value_J+","+y2
                    Px_y_dash*=CondProbTable[key1]
                else:
                    CT_Y=Attr_data[i].values_count[line[i]][1]
                    Px_y_dash*=(CT_Y+1)/(C_Y + len(Attr_data[i].values))
            result2=Px_y_dash*PY

            final_result1=Prob((result1),(result1+result2))
            final_result2=Prob((result2),(result1+result2))
            if final_result1>final_result2:
                if line[-1]==Attr_data[-1].values[0]:
                    correct_classified+=1
                temp = line[-1]
                temp = temp.strip("'")
                print y1,temp,"%.12f"%final_result1
            else:
                if line[-1]==Attr_data[-1].values[1]:
                    correct_classified+=1
                temp = line[-1]
                temp = temp.strip("'")
                print y2,temp,"%.12f"%final_result2
    print correct_classified,count

def LCurve(TrainFile,TestFile):
    global randlist
    # randlist=[0]*100
    # for i in range(1,101):
    #     randlist[i-1]=i
    # randlist = random.sample(randlist,25)
    InputPares(TrainFile)
    # NBClassifier(TestFile)
    TANClassifier(TestFile)

TrainFile = "vote_train.arff"
TestFile = "vote_test.arff"
# InputPares(TrainFile)
# NBClassifier(TestFile)
# TANClassifier()
# data_dir="/u/m/u/mushahid/PycharmProjects/NB_TAN_Classifier/"
# data= converters.load_any_file(data_dir+TestFile)
# data.class_is_last()
# print data
LCurve(TrainFile,TestFile)