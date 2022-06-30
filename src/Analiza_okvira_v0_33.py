'''Zadaca ovog modula je da kreira sve potrebne objekte iz tekstualne datoteke te da to proslijedi dalje, nekom centralnom programu za proracun.
Glavni dio ovog modula treba oblikovati u jednu funckiju reprezentativnog naziva - tipa, .start ili tako nesto kojoj ce se proslijedit neki osnovni podaci kao sto je path do datoteke.
Tako se moze iskoristiti ovaj program kao modul programa koji rjesava problem.'''


import math as m
import numpy as np
import numpy.polynomial.polynomial as nppp
import copy as c
import scipy.optimize as sco
from typing import List, Dict

PI=m.pi


class Node():

    '''Node class that is used for creating nodes of an structure'''

    def __init__(self,node_name:str,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=node_name
        x=float(line_list[1])
        y=float(line_list[2])
        self.coords=np.array([x,y])
        self.phi=0


class Section():

    '''Super class which only objective is to establish communcation between property and section subclasess'''

    def __init__(self,line_list:List[str]):
        self.ID=line_list[0]
        self.name=line_list[2]
        self.parameters=[]
        self.Iy=0
        self.A=0
        self.Wy=0
        self.is_there_dict_o_cons:bool=False

    def calculate(self):
        pass

    def optim(self):
        pass

    def ratio_constraint(self,list_o_elements):

        if (not self.is_there_dict_o_cons):
            self.dict_o_cons:Dict={}
            self.is_there_dict_o_cons=True

        dict_key=str(list_o_elements[0])
        dict_value=(list_o_elements[1],list_o_elements[2])
        self.dict_o_cons[dict_key]=dict_value


class I_Beam(Section):

    '''Class that creates I profile beam and is able to compute section values: area and moment of inertia'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[2]
        h=float(line_list[3])
        t=float(line_list[4])
        w1=float(line_list[5])
        t1=float(line_list[6])
        w2=float(line_list[7])
        t2=float(line_list[8])
        self.parameters=[h,t,w1,t1,w2,t2]
        self.is_there_dict_o_cons:bool=False
        self.calculate()

    def calculate(self):

        '''Method of I_beam class class that calculates all the necessary section values.'''

        h=float(self.parameters[0])
        t=float(self.parameters[1])
        w1=float(self.parameters[2])
        t1=float(self.parameters[3])
        w2=float(self.parameters[4])
        t2=float(self.parameters[5])

        A1 = w1*t1
        A2 = w2*t2
        A3 = h*t
        
        self.A = A1 + A2 + A3
        
        y1 = t1/2
        y2 = h+t1+t2/2
        y3 = h/2+t1
        yc = (A1*y1 + A2*y2 + A3*y3)/self.A
        
        Iy1 = w1*t1**3/12 + A1*(yc-y1)**2
        Iy2 = w2*t2**3/12 + A2*(yc-y2)**2
        Iy3 = (t*h**3)/12 + A3*(yc-y3)**2
        
        self.Iy = Iy1 + Iy2 + Iy3

        z_max = max([yc, h+t1+t2-yc])
        self.Wy = self.Iy / z_max

    def optim(self, list_o_param):

        '''Method that memorizes boundaries for parameters of a section'''
        for i in list_o_param:

            index=list_o_param.index(i)
            list_o_param[index]=float(i) #zasto je tu stajao int? da zaokruzi na cijelu vrijednost? nepotrebno!

        b1=(list_o_param[0],list_o_param[1])
        b2=(list_o_param[2],list_o_param[3])
        b3=(list_o_param[4],list_o_param[5])
        b4=(list_o_param[6],list_o_param[7])
        b5=(list_o_param[8],list_o_param[9])
        b6=(list_o_param[10],list_o_param[11])

        self.bounds=(b1, b2, b3, b4, b5, b6)

##    def create_section_constraints(self, line_list:List[str]):
##
##        if (line_list[0]=='BF/TF') | (line_list[0]== 'HW/TW') | (line_list[0]=='BF/HW') | (line_list[0]=='TF/TW'):
##            dict_key = line_list[0]
##        lb=line_list[1]
##        ub=line_list[2]
##        dict_value = tuple(float(lb),float()]



class C_Beam(Section):

    '''Class that creates C profile beam and is able to compute section values: area and moment of inertia'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[2]
        h=float(line_list[3])
        w=float(line_list[4])
        ts=float(line_list[5])
        tp=float(line_list[6])
        self.parameters=[h,w,ts,tp]
        self.is_there_dict_o_cons:bool=False
        self.calculate()

    def calculate(self):

        h=float(self.parameters[0])
        w=float(self.parameters[1])
        ts=float(self.parameters[2])
        tp=float(self.parameters[3])

        A1 = (w-ts)*tp
        A2 = h*ts
        self.A = 2*A1 + A2
        self.Iy = (w-ts)*tp**3/12 + A1*((h-tp)/2)**2 + ts*h**3/12
        self.Wy = 2*self.Iy/h

    def optim(self, list_o_param):

        '''Method that memorizes boundaries for parameters of a section'''

        for i in list_o_param:

            index=list_o_param.index(i)
            list_o_param[index]=int(i)

        b1=(list_o_param[0],list_o_param[1])
        b2=(list_o_param[2],list_o_param[3])
        b3=(list_o_param[4],list_o_param[5])
        b4=(list_o_param[6],list_o_param[7])

        self.bounds=(b1, b2, b3, b4)

class CircBar(Section):

    '''Class that creates circular bar beam and is able to compute section values: area and moment of inertia'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[2]
        d=float(line_list[3])
        self.parameters=[d]
        self.is_there_dict_o_cons:bool=False
        self.calculate()

    def calculate(self):

        d=float(self.parameters[0])

        self.A = d**2*PI/4
        self.Iy = d**4*PI/64
        self.Wy = 2*self.Iy/d

    def optim(self, list_o_param):

        '''Method that memorizes boundaries for parameters of a section'''

        for i in list_o_param:

            index=list_o_param.index(i)
            list_o_param[index]=int(i)

        b1=(list_o_param[0],list_o_param[1])

        self.bounds=b1

class CircTube(Section):

    '''Class that creates circular tube beam and is able to compute section values: area and moment of inertia'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[2]
        du=float(line_list[3])
        dv=float(line_list[4])
        self.parameters=[du,dv]
        self.is_there_dict_o_cons:bool=False
        self.calculate()

    def calculate(self):

        du=float(self.parameters[0])
        dv=float(self.parameters[1])

        self.A = (dv**2-du**2)*PI/4
        self.Iy = (dv**4-du**4)*PI/64
        self.Wy = 2*self.Iy/dv

    def optim(self, list_o_param):

        '''Method that memorizes boundaries for parameters of a section'''

        for i in list_o_param:

            index=list_o_param.index(i)
            list_o_param[index]=int(i)

        b1=(list_o_param[0],list_o_param[1])
        b2=(list_o_param[2],list_o_param[3])

        self.bounds=(b1, b2)

class Rectangle(Section):

    '''Class that creates rectangular tube beam and is able to compute section values: area and moment of inertia'''

    def __init__(self,line_list:List[str]):

        self.ID_sec=int(line_list[0])
        self.name=line_list[2]
        hv=float(line_list[3])
        wv=float(line_list[4])
        hu=float(line_list[5])
        wu=float(line_list[6])
        self.parameters=[hv,wv,hu,wu]
        self.is_there_dict_o_cons:bool=False
        self.calculate()

    def calculate(self):


        hv=float(self.parameters[0])
        wv=float(self.parameters[1])
        hu=float(self.parameters[2])
        wu=float(self.parameters[3])

        self.A=hv*wv-hu*wu
        self.Iy=wv*hv**3/12-wu*hu**3/12
        self.Wy=2*self.Iy/hv

    def optim(self, list_o_param):

        '''Method that memorizes boundaries for parameters of a section'''

        for i in list_o_param:

            index=list_o_param.index(i)
            list_o_param[index]=int(i)

        b1=(list_o_param[0],list_o_param[1])
        b2=(list_o_param[2],list_o_param[3])
        b3=(list_o_param[4],list_o_param[5])
        b4=(list_o_param[6],list_o_param[7])

        self.bounds=(b1, b2, b3, b4)

class TableSection(Section):    #TAKVI SE SECTONI ne mogu optimizirati 

    '''Class that creates table section. Table sections are provided with area and moments of inertia at initializing. Technical manuals often incorporate that data. THEY DON'T HAVE PARAMETERS THAT CAN'T BE CHANGED'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[2]
        self.A=float(line_list[3])
        self.Iy=float(line_list[4])
        self.Wy=float(line_list[5])


class Material():

    '''Class that is used to define linear-elastic material.'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[1]
        self.E=float(line_list[2])
        self.Poiss=float(line_list[3])
        self.dens=float(line_list[4])
        self.sigmadop=float(line_list[5])


class Property():

    '''Class that creates properties: combinations of sections and material'''

    def __init__(self,line_list:List[str], materials, sections):

        material=int(line_list[2])
        section=int(line_list[3])

        self.ID=int(line_list[0])
        self.name=line_list[1]
        self.mat=materials[material-1]          #ovaj ...-1 dolazi jer je zasad organizirano ovako: u ulaznoj datoteci pise ID. ID je povezan zasad s listom objekata tako da je indeks list (ID-1).
        self.sect=sections[section-1]           #material i section su varijable koje imaju vrijednost INDEKSA u listi objekata.



class Beam():

    def __init__(self,beam_name,numer_beam,line_list:List[str], structure_obj, ):


        node_num1 = int(line_list[1])             #Ovo su cjelobrojni indeksi odgovarajucih objekata u njihovim listama: nodes i properties listama
        node_num2 = int(line_list[2])
        prop = int(line_list[3])

        self.name = beam_name                     #Name se mozda moze koristiti i u vizualizaciji - nazivi greda pojedinih.
        self.node1 = structure_obj.nodes[node_num1-1]           #Zasad je slozeno u strukturi pamcenje pozicije preko ID-a pojedinog node-a. To znaci da se id treba dodijeljivat automatski.. od 1 do kolko ih ima... i da to korespondira s polozajem u listi.
        self.node2 = structure_obj.nodes[node_num2-1]
        self.length = self.length_calc(self.node1,self.node2)
        self.m12 = 0
        self.m21 = 0
        self.prop = structure_obj.properties[prop-1]
        self.calculate_k_ij()
        num_o_fields = m.ceil(self.length/structure_obj.kd)             #num_o_fields - broj polja na koja je podijeljena greda, vezano uz duljinu vektora self.intrinsic_diagram. kd - korak diskretizacije"
        self.kd_local = self.length/num_o_fields          #kd_local - lokalni korak diskretizacije"
        self.x = np.array(np.arange(0,self.length+self.kd_local,self.kd_local))   #np.arrange se mora koristiti za kreiranje takvog niza. Range može samo cjelobrojne brojeve imati za argumente. +self.kd_local"
        self.intrinsic_diagram = np.zeros(len(self.x))
        self.intrinsic_diagram_w_trap:np.ndarray = np.zeros(len(self.x))
        
        self.max_moment_independently = 0
        
        self.max_s = 0
        self.y_cg = None

    @property
    def sigma_limit(self):
        return self.prop.mat.sigmadop

    def max_stress_of_intrinsic_diagram(self):
        return np.abs(self.intrinsic_diagram).max

    def get_stress_over_limit(self):
        
        return self.max_s/self.prop.mat.sigmadop

    def length_calc(self,node1:Node,node2:Node) -> float:

        node_vector = node2.coords-node1.coords
        length_o_beam = np.linalg.norm(node_vector)

        return float(length_o_beam)

    def calculate_k_ij(self):

        self.k_ij:float = 2*self.prop.mat.E*self.prop.sect.Iy/self.length

    def create_load(self,line_list:List[str]):

        load_type = line_list[1]
        value = float(line_list[3])

        if load_type == "F":
            placement = float(line_list[4])
            m12 = -value*placement*self.length*(1-placement)**2
            m21 = value*(1-placement)*self.length*(placement)**2          #FORMULE IZ IP1 PRIRUCNIKA, STR. 523.

        elif load_type=="q":                                        #Jedinica sile / jedinica duljine
            m12 = -value*(self.length**2)/12
            m21 = -m12

        elif load_type=="qlinl":                                    #trokutasta raspodjela opterecenja se spusta slijeva nadesno
                m12 = -value*(self.length**2)/20
                m21 = value*(self.length**2)/30

        elif load_type=="qlinr":                                    #trokutasta raspodjela opterecenja se spusta zdesna nalijevo
                m12 = -value*(self.length**2)/30
                m21 = value*(self.length**2)/20

        elif load_type=="M":                                        #MOGUCE DODATI JOS OPCIJA?
            m12 = value/2
            m21 = m12

        elif load_type=="anal":
            m = line_list[3]

        self.m12 = self.m12+m12
        self.m21 = self.m21+m21

        self.moment_diagram(load_type,value)

    def moment_diagram(self,load_type,value):                 #DVAPUT SE RADI ISTA PROVJERA, NEPOTREBNO! PREBACITI U CREATE_LOAD_AND_DIAGRAM - JER, MOMENTNI DIJAGRAM (OSIM MOMENATA UPETOSTI KOJI SE NE ODREÐUJU OVDJE OSTAJE VAZDA ISTI
                                                        #I KROZ OCEKIVANE PROMJENE PRESJEKA. TAKO, JEDNOM KREIRANI OSTAJE, ISTI, PA JE TE VEKTORE JEDNOSTAVNO POTREBNO ZAPAMTIT UNUTAR BEAM-A - ALI BEZ TRAPEZA OD MOMENATA UPETOSTI.
                                                         #PRI PROMJENAMA PRESJEKA TRAPEZ CE SE MIJENJATI KAKO CE VEC KOJA GREDA IMATI UDJELA U KRUTOSTI U CVORU

        if load_type=="F":

            placement = float(line_list[4])
            peak_loc = 1+m.floor(placement*self.length/self.kd_local)         #broj koraka diskretizacije do vrha mometnog dijagrama od konc. sile"
            x1 = self.x[0:peak_loc]                                  #ovo se zove: slicing an array. Potrebno je jer postoje dva pravca, dakle dvije domene."
            x2 = self.x[peak_loc:]
            moment_diag1 = np.zeros(len(x1))
            moment_diag2 = np.zeros(len(x2))
            moment_diag1 += (1-placement)*value*x1
            moment_diag2 += placement*value*(self.length-x2)
            self.intrinsic_diagram += np.concatenate((moment_diag1,moment_diag2), axis = None)

        elif load_type=="q":

            self.intrinsic_diagram +=   value/2*self.x**2 - value*self.length/2*self.x   #parabola koja opisuje momentni dijagram"

        elif load_type=="qlinr":

            self.intrinsic_diagram +=  1/6*value/self.length*self.x**3 - 1/6*value*self.length*self.x #ovo provjeriti integracijom! 

        elif load_type=="qlinl":

            moment_diag = np.zeros(len(self.x))
            moment_diag +=  1/6*value/self.length*self.x**3 - 1/6*value*self.length*self.x  #izracun na isti nacin kao i qlinr, samo koristenje numpy.flip da zamijeni vrijednosti oko y osi
            moment_diag = np.flip(moment_diag)
            self.intrinsic_diagram += moment_diag


        elif load_type=="M":

            placement = float(line_list[4])
            peak_loc = 1+m.floor(placement*self.length/kd_local)
            x1 = self.x[0:peak_loc]
            x2 = self.x[peak_loc:]
            moment_diag1 = np.zeros(len(x1))
            moment_diag2 = np.zeros(len(x2))
            moment_diag1 += -value*x1/self.length
            moment_diag2 += value*(self.length-x2)/self.length
            self.intrinsic_diagram += np.concatenate((moment_diag1,moment_diag2), axis = None)

        elif load_type=="anal": #NOT IMPLEMENTED
            pass

    def moment_diagram_clear(self): #U slucaju neke potrebe mozda ovako nesto napravit. VRLO VJEROJATNO NECE BITI POTREBNO

        self.intrinsic_diagram = np.zeros(num_o_fields+1)

    def max_stress(self):

##        trapezius = np.linspace(self.M12, -self.M21,len(self.x))    #racunanje trapeza zbog nejednakih momentata upetosti u cvorovima
##        self.intrinsic_diagram_w_trap = self.intrinsic_diagram + np.ravel(trapezius)             #np.ravel funkcija potrebna da se mogu zbrojiti dva niza - jer su inace drugacijih oblika (R,1) i (R,)
##
##        
##        trapezius = np.linspace(self.m12-self.M12, self.m21-self.M21, len(self.x))
##        self.intrinsic_diagram_w_trap = self.intrinsic_diagram - np.ravel(trapezius)             #np.ravel funkcija potrebna da se mogu zbrojiti dva niza - jer su inace drugacijih oblika (R,1) i (R,)
##
##        max_moment = np.abs(self.intrinsic_diagram_w_trap).max()
        
        moment_node1 = abs(self.M12)
        moment_node2 = abs(self.M21)

        possible_maxs = [ moment_node1, moment_node2 ]
        max_moment = float(max(possible_maxs))
        
        self.max_s = abs(max_moment/self.prop.sect.Wy)                 #izracun maksimalnog naprezanja na gredi


class Structure():

    '''Supervising and super class that represents all of the model: it's nodes, sections, materials, properties, beams and loads assigned to beams. It's purpose is to establish
        communication between different objects and to run the analysis. It has access to all needed parts for optimization.'''

    def __init__(self, kd):

        self.nodes:List[Node] = []
        self.materials:List[Material] = []
        self.sections:List[Section] = []
        self.properties:List[Property] = []
        self.beams:List[Beam] = []

        self.kd = kd
        y_cgs_calculated = False              #CONTROL PARAMETER FOR CALCULATION OF VERTICAL POSITION OF CENTER OF GRAVITY
        self.first_evaluation_of_global_equation = True
        self.m=[]                               #Vector of moments due to load (individual beam)
        self.y_cgs_calculated = False


    @property
    def beam_stresses_array(self) -> List[float]:
        
        stresses = []
        for beam in self.beams:
            stresses.append(beam.max_s)

        return stresses

    def calculate_m_vector(self):

        self.m = np.zeros((len(self.nodes),1))
        
        for beam in self.beams:
            self.m[beam.node1.ID-1] += (-beam.m12)      #VEKTOR MOMENATA (zapravo se ne treba kreirati iznova svaki puta, tu se moze ustedjeti na racunalnom vremenu - isto kad bude moguce vise load caseova! Za svaki load case ovo izracunati)
            self.m[beam.node2.ID-1] += (-beam.m21)        

    def global_equation(self)-> np.ndarray: #vraca matricu nxn, gdje je n broj mogucih kuteva zakreta

        '''"Method that sets global system of linear algebraic equations for calculation of angular displacements based on compatibility conditions and equilibrium equations of every nodes.'''

        phi = np.zeros((len(self.nodes),1))  #inicijalizacija prazne matrice

        K = np.zeros((len(self.nodes),len(self.nodes)))
        
        if self.first_evaluation_of_global_equation == True:
            self.calculate_m_vector()
            self.first_evaluation_of_global_equation = False
    
        for beam in self.beams:
            
                    beam.prop.sect.calculate()
                    beam.calculate_k_ij()
                    ixgrid = np.ix_([beam.node1.ID-1, beam.node2.ID-1], [beam.node1.ID-1, beam.node2.ID-1])   #Koristenje np.ix_ da se na prava mjesta u globalnoj matrici doda lokalna krutost."
                    K[ixgrid] += np.array([[2*beam.k_ij, beam.k_ij], [beam.k_ij, 2*beam.k_ij]])             #numpy zbrajanje preko submatrica  polje indeksa... np.ix_ numpy je brzi - da wrapped FORTRAN C++"

##                    self.m[beam.node1.ID-1] += (-beam.m12)     #Vektor "malih" m_ij momenata! Zamijenjeno sa funkcijom calculate_m_vector
##                    self.m[beam.node2.ID-1] += (-beam.m21)

        

        K_inv = np.linalg.inv(K)

##        #ISPIS I RACUNANJE UVJETOVANOSTI RADI KONTROLE."
##        
##        uvjetovanost = np.linalg.norm(K,'fro')*np.linalg.norm(K_inv,'fro')       
##
##        print("Uvjetovanost matrice K iznosi: \n", uvjetovanost)
##        print(f'Matrica K: \n, {K[0:5,0:5]*1e-10}')
##        print(f'Inverz matrice K: \n {K_inv[0:5,0:5]*1e+12}')
##        print(f'Vektor momenata:\n {self.m}')
##        print('')


        phi = np.matmul(K_inv, self.m)       #MATRICNO MNOZENJE

        phi = np.ravel(phi)

        return phi



    def calculate_all(self):

        '''Method that based that calls function that calculates angular displacements, assigns those displacements to appropriate nodes and then calculates moments at the end of beams.'''

        phi = self.global_equation()
##        print(f'Phi: \n {phi}')

        #Pripisujemo cvorovima njihove kuteve zakreta

        for i in range(0,len(self.nodes)):
            self.nodes[i].phi = phi[i]        #POJASNJENJE: U cvoru, grede se zakrecu zajedno z00a isti kut. Primjetimo, ova metoda uzima u obzir krutosti. Tako da ce taj zakret biti
                                            #najblizi zakretu najkruce grede u stvarnosti. VAZNO - prema redoslijedu cvorova su i kreirane jednadzbe, pa tako ce i rjesenja biti ispravno poredana


        #Momenti upetosti na kraju cvorova

        for beam in self.beams:
            
            beam.M12 = beam.k_ij*(2*beam.node1.phi + beam.node2.phi)+beam.m12
            beam.M21 = beam.k_ij*(2*beam.node2.phi + beam.node1.phi)+beam.m21
            beam.max_stress()

    def change_opt_param(self):

        '''Method that changes parameters that is called after optimization step, calls for calculation of section parameters of beams and then calls for calculation of global equation again - Structure.calculate_all.
            Not yet implemented. There wasn't a need for it.'''

        pass

    def get_mass(self) -> float:

        '''Method that can be used by other programs in order to fetch (get) mass of a structure in order to form objective function'''

        mass = 0
        
        for beam in self.beams:
            
            mass += beam.length*beam.prop.sect.A*beam.prop.mat.dens

        return mass

    def calculate_y_cgs(self):

        '''Method that calculates y_cg of beams - that is - vertical position of each beams center of gravity. That doesn't changes, so it only needs to be calculated once!'''

        for beam in self.beams:
            beam.y_cg = ( beam.node1.coords[1] + beam.node2.coords[1] ) / 2

    def get_vertical_CG_position(self) -> float:

        '''Method that can be invoked to calculate vertical position of the center of gravity point'''

        CG_position = 0

        if self.y_cgs_calculated == False:
            self.calculate_y_cgs()
            self.y_cgs_calculated = True
            
        for beam in self.beams:

            #CALCULATING numerator of formula CG = sum(beam.y_cg*beam.length*beam.prop.sect.A*beam.prop.mat.dens for beam in beams) / Structure.get_mass()
            CG_position += beam.y_cg * beam.length * beam.prop.sect.A * beam.prop.mat.dens

        CG_position = CG_position/self.get_mass() #KAKO NAPRAVITI DA PRORACUN MASE NEJDE PONOVNO? NEGO AKO JE VEC IZRACUNATA U OVOM KRUGU, DA JEDNOSTAVNO JU ISKORISTI? I KAK ZNATI DA JE TO AKTUALNA MASA, DA JE UPDATEA-ANA?
            

        return CG_position

    def get_section_parameters(self, section_ID) -> List:

        '''Method that returns section array in order to facilitate extraction of parameters array for pymoo optimization. '''
        
        return self.sections[section_ID-1].parameters

class Input_file():

    '''Class that stores data about input file used to generate analysis as well as functionality that goes with it.'''

    def __init__(self,path, kd):
        self.path = path
        self.kd = kd

    def load_file(self):

        try:
            self.inp_file = open(self.path,"rt")
        except:
            print("Greska. Ili datoteka ne postoji, ili je unesena putanja kriva.")
            print("Provjeriti ispravnost putanje i postojanje datoteke!")

    def create_model(self):

        # DEFINICIJA NUKMERATORA ID-a ZA STVARANJE DEFAULTNIH IMENA AKO NE POSTOJE

        numer_node = 1
        numer_mat = 1
        numer_sect = 1
        numer_prop = 1
        name_prop = 1
        numer_beam = 1
        name_beam = 1

        structure_obj = Structure(self.kd)

        #DEFINIRANJE NEKIH GLOBALNIH BROJACA I LISTA

        tpi = 0                       #text position index
        lines = []                   #buduce linije tekstualne datoteke

        #PROLAZENJE KROZ TEKSTUALNU DATOTEKU I ZATVARANJE DATOTEKE

        for line_string in self.inp_file:
            lines.append(line_string)               #lines sada sadrze sav tekst tekstualne datoteke
        self.inp_file.close()

        #CITANJE BROJA OBJEKATA I KREIRANJE ISTIH POZIVANJEM FUNKCIJA

        quantities_read = False
        curr = 0
        num_o_objects = []
        sections_to_opt = []
        num_o_param = []

        while tpi<len(lines):

            if lines[tpi][0]!="#" and not quantities_read:  #u num_o_objects se pamti broj objekata - cvorova, presjeka, greda...
                quantities_read = True
                line_string = lines[tpi]                      #Prvih linija komentara moze biti vise, pa kad naide na liniju bez prvog znaka # - cita koliko ima kojih objekata
                line_string = line_string.strip()
                line_list = line_string.split(",")
                i = 0
                for num in line_list:
                    line_list[i] = int(num)
                    i += 1
                num_o_objects = line_list

                #print(num_o_objects)                        #RADI KONTROLE

            elif lines[tpi][0]!="#" and quantities_read:

                if curr==0:                                 #NODE

                    for i in range(0,num_o_objects[curr]):

                        line_string = lines[tpi]
                        line_list = self.word_splitting(line_string)
                        numer_node, structure_obj = self.node_creation(line_list, numer_node, structure_obj)
                        tpi += 1

                elif curr==1:                                 #MATERIAL

                    for i in range(0,num_o_objects[curr]):

                        line_string = lines[tpi]
                        line_list = self.word_splitting(line_string)
                        numer_mat, structure_obj = self.material_creation(line_list, numer_mat, structure_obj)
                        tpi += 1

                elif curr==2:                                 #SECTION

                    for i in range(0,num_o_objects[curr]):

                        line_string = lines[tpi]
                        line_list = self.word_splitting(line_string)
                        numer_sect, structure_obj = self.section_creation(line_list, numer_sect, structure_obj)
                        tpi += 1

                elif curr==3:                                 #PROPERTY

                    for i in range(0,num_o_objects[curr]):

                        line_string = lines[tpi]
                        line_list = self.word_splitting(line_string)
                        numer_prop, structure_obj = self.property_creation(line_list, numer_prop, structure_obj)
                        tpi += 1

                elif curr==4:                                 #BEAM

                    for i in range(0,num_o_objects[curr]):

                        line_string = lines[tpi]
                        line_list = self.word_splitting(line_string)
                        name_beam, numer_beam, structure_obj = self.beam_creation(line_list, name_beam, numer_beam, structure_obj)
                        tpi += 1

                elif curr==5:                                 #LOAD

                    for i in range(0,num_o_objects[curr]):

                        line_string = lines[tpi]
                        line_list = self.word_splitting(line_string)
                        structure_obj = self.load_creation(line_list, structure_obj)
                        tpi += 1

                curr += 1

            tpi  +=  1

        return structure_obj


    def word_splitting(self, line_string:str)->List[str]:

        '''Function used for stripping and splitting words from textual file. Mainly used in creation of instances of a class.'''

        line_string = line_string.strip()     #cisti pocetak od razmaka

        line_list = line_string.split(",")    #razdvaja na pojedine rijeci

        i = 0
        for word in line_list:
            word = word.strip()
            word = word.strip('"')
            line_list[i] = word
            i += 1

        return line_list

    #CREATION FUNCTIONS - razne funkcije za kreiranje niza objekata na globalnoj razini...

    def node_creation(self, line_list:List[str], numer_node, structure_obj):

        '''Function that creates NODE OBJECTS from input file.'''

        node_name = "Node"+str(numer_node)                                    #Automatsko dodavanje imena cvoru
        structure_obj.nodes.insert(numer_node, Node(node_name,line_list))    #KREIRANJE CVORA - staviti metodu append

        numer_node += 1                   #Povecavanje brojaca za 1
        
        return numer_node, structure_obj


    def material_creation(self, line_list, numer_mat, structure_obj):

        '''Function that creates MATERIAL OBJECTS from input file.'''
        structure_obj.materials.insert(numer_mat,Material(line_list)) #pogledati datoteku sto je koji element niza line_list
        
        numer_mat += 1

        return numer_mat, structure_obj


    def beam_creation(self, line_list, name_beam, numer_beam, structure_obj):   #detaljnije pojasnjenje strukture - vidi u node_creation funkciji. Tamo je opisan pojedini redak - ovdje se ponavlja...

        '''Function that creates BEAM OBJECTS from input file.'''


        beam_name = "Beam"+str(name_beam)                                         # Vidjeti je li potrebno??? Moze bit, ali jedino ako se uz to omoguci unos kao dodatna mogucnost.
        structure_obj.beams.insert(numer_beam,Beam(beam_name,numer_beam,line_list, structure_obj))

        name_beam += 1
        numer_beam += 1

        return name_beam, numer_beam, structure_obj


    def section_type(self, line_list, numer_sect, structure_obj):

        '''Function that is actually a switch-case. Dependent on the read type of the section, it instatiates the right one'''

        chooser = line_list[1]

        if chooser=="I_Beam":
            structure_obj.sections.insert(numer_sect,I_Beam(line_list))
        elif chooser=="T_Beam_on_plate":
            structure_obj.sections.insert(numer_sect,T_Beam_on_plate(line_list))
        elif chooser=="C_Beam":
            structure_obj.sections.insert(numer_sect,C_Beam(line_list))
        elif chooser=="CircBar":
            structure_obj.sections.insert(numer_sect,CircBar(line_list))
        elif chooser=="CircTube":
            structure_obj.sections.insert(numer_sect,CircTube(line_list))
        elif chooser=="Rectangle":
            structure_obj.sections.insert(numer_sect,Rectangle(line_list))
        elif chooser=="TableSection":
            structure_obj.sections.insert(numer_sect,TableSection(line_list))

        return structure_obj

    def section_creation(self, line_list, numer_sect, structure_obj):

        '''Function that creates SECTION OBJECTS from input file.'''

        structure_obj = self.section_type(line_list, numer_sect, structure_obj)
        numer_sect += 1

        return numer_sect, structure_obj


    def property_creation(self, line_list, numer_prop, structure_obj):

        '''Function that creates PROPERTY OBJECTS from input file by assigning section and material to it. Also, takes parameter called "cost".
            That way, it is predicted that cost optimization is possible to achieve.'''

        structure_obj.properties.insert(numer_prop, Property(line_list, structure_obj.materials, structure_obj.sections))
        numer_prop += 1

        return numer_prop, structure_obj


    def load_creation(self, line_list, structure_obj):

        '''Function that calls proper beam object and creates loads that act on it'''

        section_ID = int(line_list[2])
        structure_obj.beams[section_ID-1].create_load(line_list)

        return structure_obj


    def switch_type(self, line_list):

        '''Function that calles appropriate function for creation of objects from input file.'''
        chooser = line_list[0]
        if chooser=="Node":
            node_creation(line_list)
        elif chooser=="Material":
            material_creation(line_list)
        elif chooser=="Section":
            section_creation(line_list)
        elif chooser=="Property":
            property_creation(line_list)
        elif chooser=="Load":
            load_creation(line_list)
        elif chooser=="Beam":
            beam_creation(line_list)

class Constraint_stress():

    def __init__(self,beam:Beam):

        self.beam = beam

    def constraint(self,x) -> float: #x je tu samo radi zahtjeva scipy funkcija

        sigmadop = self.beam.prop.mat.sigmadop
        max_stress = self.beam.max_s

        return sigmadop-max_stress


def calculate_problem(structure_obj:Structure):

    '''Method used for optlib_pymoo.py for calling of calculation of already initialized problem. Problem is initialized by loading data from input text file.
    and is then connected via optlib_pymoo.py to a pymoo optimization module'''

    structure_obj.calculate_all()

def get_stress_cons_lt_0() -> List[float]:

    '''Method for pymoo and perhaps some other optimization modules that take list of values from evaluated constraints. For structural frame analysis it is common to compare beam stresses to allowable values for material.
        Results are returned in a list (pymoo) to be saved in "out" dictionary like this: out['G']=ao.get_stress_cons_lt_0() where ao is from: import Analiza_okvira as ao. '''

    beam_stress_cons = []

    for beam in structure_obj.beams:
        con_value = beam.max_s - beam.mat.sigmadop
        beam_stress_cons.append(con_value)

        return beam_stress_cons

def get_stress_cons_gt_0() -> List[float]:

    beam_stress_cons = []

    for beam in structure_obj.beams:
        con_value = beam.max_s - beam.mat.sigmadop
        beam_stress_cons.append(con_value)

        return beam_stress_cons
