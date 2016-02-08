from __future__ import division
import re

Attributelist=[]
TrainDataSet=[]
attributeCounter=0
ComputerWieghts={}
Mutualinfo={}
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



def InputPares():
    globals()
    finput= open('lymph_train.arff','r')
    # lines = [line.rstrip('\n') for line in finput]
    for line in finput:
        testlist=[1,2,3,4]
        line=line.rstrip('\n')
        if line.startswith('@attribute'):
            templist =  line.split(' ')
            values = re.findall(r'\{([^]]*)\}',line)
            newlist=values[0].strip("'").split(",")
            newlist = [each.strip() for each in newlist]
            templist[1]=templist[1].replace('\\','').replace("'",'')
            newatr = Attribute(templist[1],newlist)
            Attributelist.append(newatr)
        if line.startswith('@attribute') or line.startswith('@relation') or line.startswith('%') or line.startswith('@data'):
            continue
        else:
            line=line.strip()
            line= line.split(',')
            TrainDataSet.append(line)
            if line[-1]==Attributelist[-1].values[0]:
                for i in range(len(line)):
                    Attributelist[i].values_count[line[i]][0]+=1
            else:
                for i in range(len(line)):
                    Attributelist[i].values_count[line[i]][1]+=1
    pass

def NBClassifier():
    testinput= open('lymph_test.arff','r')
    correct_classified=0
    C_X=Attributelist[-1].values_count[Attributelist[-1].values[0]][0]
    C_Y=Attributelist[-1].values_count[Attributelist[-1].values[1]][1]
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
            Px_y=1.0
            Px_y_dash=1.0
            for i in range(len(line)-1):
                CT_Y=Attributelist[i].values_count[line[i]][0]
                Px_y*=(CT_Y+1)/+(C_X+len(Attributelist[i].values))
            result1=Px_y*PX
            for i in range(len(line)-1):
                CT_Y=Attributelist[i].values_count[line[i]][1]
                Px_y_dash*=(CT_Y+1)/(C_Y+len(Attributelist[i].values))
            result2=Px_y_dash*PY
            final_result1=float(result1)/float(result1+result2)
            final_result2=float(result2)/float(result1+result2)
            if final_result1>final_result2:
                if line[-1]==Attributelist[-1].values[0]:
                    correct_classified+=1
                print(final_result1)
            else:
                if line[-1]==Attributelist[-1].values[1]:
                    correct_classified+=1
                print(final_result2)

    print(correct_classified,count)

def CalculateConditionalMutualInfo():
    globals()
    for Att_I in Attributelist:
        if Att_I.index==Attributelist[-1].index:
            continue
        Mutualinfo[Att_I]={}
        for Att_J in Attributelist:
            if Att_J.index==Attributelist[-1].index:
                continue
            if Att_I.index==Att_J.index:
                continue
            Mutualinfo[Att_I][Att_J]={}
            for value_I in Att_I.values:
                Mutualinfo[Att_I][Att_J][value_I]={}
                for value_J in Att_J.values:
                    Mutualinfo[Att_I][Att_J][value_I][value_J]=[0,0]
                    for row in TrainDataSet:
                        if row[Att_I.index]==value_I and row[Att_J.index]==value_J and row[-1]==Attributelist[-1].values[0]:
                            Mutualinfo[Att_I][Att_J][value_I][value_J][0]+=1
                        if row[Att_I.index]==value_I and row[Att_J.index]==value_J and row[-1]==Attributelist[-1].values[1]:
                            Mutualinfo[Att_I][Att_J][value_I][value_J][1]+=1
    pass

def get_C_xi_xj_given_y(Att_I,Att_J,y):
    TotalSum=0
    for value_I in Att_I.values:
        for value_J in Att_J.values:
            TotalSum+=Mutualinfo[Att_I][Att_J][value_I][value_J][y]
    return TotalSum

def get_C_xi_xj(Att_I,Att_J,y):
    TotalSum=0
    for value_I in Att_I.values:
        for value_J in Att_J.values:
            TotalSum+=Mutualinfo[Att_I][Att_J][value_I][value_J][y]
    return TotalSum

def CalculateWeights():
    globals()
    CalculateConditionalMutualInfo()
    C_xi_xj_given_y1=get_C_xi_xj_given_y(Attributelist[0],Attributelist[1],1)
    print C_xi_xj_given_y1
    # for Att_I in Attributelist:
    #     for Att_J in Attributelist:
    #         if Att_I.name==Att_J.name:
    #             continue
    #         C_xi_xj_given_y1=get_C_xi_xj(Att_I,Att_J,0)
    #         P_xi_xj_given_y1=
    #         ComputerWieghts[Att_I.name,Att_J.name]=[]



def TANClassifier():
    CalculateWeights()


InputPares()
NBClassifier()
TANClassifier()