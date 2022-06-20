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
        self.calculate(self.parameters)

    def calculate(self,parameters):

        '''Method of I_beam class class that calculates all the necessary section values.'''

        self.parameters=parameters

        h=float(parameters[0])
        t=float(parameters[1])
        w1=float(parameters[2])
        t1=float(parameters[3])
        w2=float(parameters[4])
        t2=float(parameters[5])

        A1 = w1*t1
        A2 = w2*t2
        A3 = (h-t1-t2)*t
        self.A = A1 + A2 + A3
        y1 = t1/2
        y2 = (h+t1-t2)/2
        y3 = (h-t2/2)
        yc = (A1*y1 + A2*y2 + A3*y3)/self.A
        Iy1 = w1*t1**3/12 + A1*(yc-y1)**2
        Iy2 = w2*t2**3/12 + A2*(yc-y2)**2
        Iy3 = ((h-t1-t2)**3*t)/12 + A3*(yc-y3)**2
        self.Iy = Iy1 + Iy2 + Iy3
        self.Wy=2*self.Iy/h

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


class T_Beam_on_plate(Section):

    '''Class that creates I profile beam and is able to compute section values: area and moment of inertia'''

    def __init__(self,line_list:List[str]):

        self.ID=int(line_list[0])
        self.name=line_list[2]
        h=float(line_list[3])
        t=float(line_list[4])
        self.w_plate=float(line_list[5])        #dio oplate koji se ne optimizira
        self.t_plate=float(line_list[6])
        wf=float(line_list[7])
        tf=float(line_list[8])
        self.parameters=[h,t,wf,tf]
        self.is_there_dict_o_cons:bool=False
        self.calculate(self.parameters)

    def calculate(self,parameters):

        '''Method of I_beam class class that calculates all the necessary section values.'''

        self.parameters=parameters

        h=float(self.parameters[0])
        t=float(self.parameters[1])
        w_plate=float(self.w_plate)
        t_plate=float(self.t_plate)
        wf=float(self.parameters[4])
        tf=float(self.parameters[5])

        A1 = w_plate*t_plate
        A2 = w2*tf
        A3 = (h-t_plate-tf)*t
        self.A = A1 + A2 + A3
        y1 = t_plate/2
        y2 = (h+t_plate-tf)/2
        y3 = (h-tf/2)
        yc = (A1*y1 + A2*y2 + A3*y3)/self.A
        Iy1 = w_plate*t_plate**3/12 + A1*(yc-y1)**2
        Iy2 = w2*tf**3/12 + A2*(yc-y2)**2
        Iy3 = ((h-t_plate-tf)**3*t)/12 + A3*(yc-y3)**2
        self.Iy = Iy1 + Iy2 + Iy3
        self.Wy=2*self.Iy/h

    def optim(self, list_o_param):  

        '''Method that memorizes boundaries for parameters of a section'''
        for i in list_o_param:
            
            index=list_o_param.index(i)
            list_o_param[index]=float(i) #zasto je tu stajao int? da zaokruzi na cijelu vrijednost? nepotrebno!
        
        b1=(list_o_param[0],list_o_param[1])
        b2=(list_o_param[2],list_o_param[3])
        b3=(list_o_param[4],list_o_param[5])
        b4=(list_o_param[6],list_o_param[7])


        self.bounds=(b1, b2, b3, b4)

       
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
        self.calculate(self.parameters)

    def calculate(self,parameters):

        self.parameters=parameters

        h=float(parameters[0])
        w=float(parameters[1])
        ts=float(parameters[2])
        tp=float(parameters[3])

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
        self.calculate(self.parameters)

    def calculate(self,parameters):

        self.parameters=parameters

        d=float(parameters[0])

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
        self.calculate(self.parameters)

    def calculate(self,parameters):

        self.parameters=[du,dv]

        du=float(parameters[0])
        dv=float(parameters[1])
        
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
        self.calculate(self.parameters)

    def calculate(self,parameters):

        self.parameters=[hv,wv,hu,wu]

        hv=float(parameters[0])
        wv=float(parameters[1])
        hu=float(parameters[2])
        wu=float(parameters[3])
        
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

class TableSection(Section):    #Zasad još nije sasvim funkcionalno - takvi se section nemogu optimizirati.

    '''Class that creates table section. Table sections are provided with area and moments of inertia at initializing. Technical manuals often incorporate that data.'''

    def __init__(self,line_list:List[str]):

        self.name=line_list[0]
        self.ID=int(line_list[0])
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

    def __init__(self,line_list:List[str]):

        global materials, sections

        material=int(line_list[2])
        section=int(line_list[3])

        self.ID=int(line_list[0])
        self.name=line_list[1]
        self.mat=materials[material-1]          #ovaj ...-1 dolazi jer je zasad organizirano ovako: u ulaznoj datoteci pise ID. ID je povezan zasad s listom objekata tako da je indeks list (ID-1).
        self.sect=sections[section-1]           #material i section su varijable koje imaju vrijednost INDEKSA u listi objekata.



class Beam():

    def __init__(self,beam_name,numer_beam,line_list:List[str]):

        global structure_obj, kd

        node_num1=int(line_list[1])             #Ovo su cjelobrojni indeksi odgovarajucih objekata u njihovim listama: nodes i properties listama
        node_num2=int(line_list[2])
        prop=int(line_list[3])

        self.name=beam_name                     #Name se mozda moze koristiti i u vizualizaciji - nazivi greda pojedinih.
        self.node1=nodes[node_num1-1]           #Zasad je slozeno u strukturi pamcenje pozicije preko ID-a pojedinog node-a. To znaci da se id treba dodijeljivat automatski.. od 1 do kolko ih ima... i da to korespondira s polozajem u listi.
        self.node2=nodes[node_num2-1]
        self.length=self.length_calc(self.node1,self.node2)
        self.m12=0
        self.m21=0
        self.prop=structure_obj.properties[prop-1]
        self.calculate_k_ij()
        num_o_fields=m.ceil(self.length/kd)             #num_o_fields - broj polja na koja je podijeljena greda, vezano uz duljinu vektora self.intrinsic_diagram. kd - korak diskretizacije"
        self.kd_local=self.length/num_o_fields          #kd_local - lokalni korak diskretizacije"
        self.x=np.array(np.arange(0,self.length+self.kd_local,self.kd_local))   #np.arrange se mora koristiti za kreiranje takvog niza. Range može samo cjelobrojne brojeve imati za argumente. +self.kd_local"
        self.intrinsic_diagram=np.zeros(len(self.x))
        self.intrinsic_diagram_w_trap:np.ndarray=np.zeros(len(self.x))
        self.max_s=0

    def length_calc(self,node1:Node,node2:Node) -> float:

        node_vector=node2.coords-node1.coords
        length_o_beam=np.linalg.norm(node_vector)
        
        return float(length_o_beam)

    def calculate_k_ij(self):

        self.k_ij:float=2*self.prop.mat.E*self.prop.sect.Iy/self.length

##    def edit_properties(self):    VRLO VJEROJATNO OVO CE BITI NEPOTREBNO
##
##        pass                        #OVO TREBA DORADITI U SURADNJI S OPTIMIZACIJSKIM ALGORITMOM

    def create_load(self,line_list:List[str]):

        load_type=line_list[1]
        value=float(line_list[3])

        if load_type=="F":
            placement=float(line_list[4])
            m12=-value*placement*self.length*(1-placement)**2
            m21=value*(1-placement)*self.length*(placement)**2          #FORMULE IZ IP1 PRIRUCNIKA, STR. 523.

        elif load_type=="q":
            m12=-value*(self.length**2)/12
            m21=-m12

        elif load_type=="qlinl":                                    #trokutasta raspodjela opterecenja se spusta slijeva nadesno
                m12=-value*(self.length**2)/20
                m21=value*(self.length**2)/30
                
        elif load_type=="qlinr":                                    #trokutasta raspodjela opterecenja se spusta zdesna nalijevo
                m12=-value*(self.length**2)/30
                m21=value*(self.length**2)/20

        elif load_type=="M":                                        #MOGUCE DODATI JOS OPCIJA?
            m12=value/2
            m21=m12

        elif load_type=="anal":
            m=line_list[3]

        self.m12=self.m12+m12
        self.m21=self.m21+m21

        self.moment_diagram(load_type,value)

    def moment_diagram(self,load_type,value):                 #DVAPUT SE RADI ISTA PROVJERA, NEPOTREBNO! PREBACITI U CREATE_LOAD_AND_DIAGRAM - JER, MOMENTNI DIJAGRAM (OSIM MOMENATA UPETOSTI KOJI SE NE ODREÐUJU OVDJE OSTAJE VAZDA ISTI
                                                        #I KROZ OCEKIVANE PROMJENE PRESJEKA. TAKO, JEDNOM KREIRANI OSTAJE, ISTI, PA JE TE VEKTORE JEDNOSTAVNO POTREBNO ZAPAMTIT UNUTAR BEAM-A - ALI BEZ TRAPEZA OD MOMENATA UPETOSTI.
                                                         #PRI PROMJENAMA PRESJEKA TRAPEZ CE SE MIJENJATI KAKO CE VEC KOJA GREDA IMATI UDJELA U KRUTOSTI U CVORU

        if load_type=="F":

            placement=float(line_list[4])
            peak_loc=1+m.floor(placement*self.length/self.kd_local)         #broj koraka diskretizacije do vrha mometnog dijagrama od konc. sile"
            x1=self.x[0:peak_loc]                                  #ovo se zove: slicing an array. Potrebno je jer postoje dva pravca, dakle dvije domene."
            x2=self.x[peak_loc:]
            moment_diag1=np.zeros(len(x1))
            moment_diag2=np.zeros(len(x2))
            moment_diag1+=(1-placement)*value*x1
            moment_diag2+=placement*value*(self.length-x2)
            self.intrinsic_diagram+=np.concatenate((moment_diag1,moment_diag2), axis=None)

        elif load_type=="q":

            self.intrinsic_diagram+=-value/2*self.x**2+value*self.length/2*self.x   #parabola koja opisuje momentni dijagram"

        elif load_type=="qlinr":

            self.intrinsic_diagram+=-1/6*value/self.length*self.x**3+1/6*value*self.length*self.x

        elif load_type=="qlinl":

            moment_diag=np.zeros(len(self.x))
            moment_diag+=-1/6*value/self.length*self.x**3+1/6*value*self.length*self.x  #izracun na isti nacin kao i qlinr, samo koristenje numpy.flip da zamijeni vrijednosti oko y osi
            moment_diag=np.flip(moment_diag)
            self.intrinsic_diagram+=moment_diag


        elif load_type=="M":

            placement=float(line_list[4])
            peak_loc=1+m.floor(placement*self.length/kd_local)
            x1=self.x[0:peak_loc]
            x2=self.x[peak_loc:]
            moment_diag1=np.zeros(len(x1))
            moment_diag2=np.zeros(len(x2))
            moment_diag1+=-value*x1/self.length
            moment_diag2+=value*(self.length-x2)/self.length
            self.intrinsic_diagram+=np.concatenate((moment_diag1,moment_diag2), axis=None)

        #elif load_type=="anal":

    def moment_diagram_clear(self): #U slucaju neke potrebe mozda ovako nesto napravit. VRLO VJEROJATNO NECE BITI POTREBNO

        self.intrinsic_diagram=np.zeros(num_o_fields+1)

    def max_stress(self):

        trapezius=np.linspace(self.M12,self.M21,len(self.x))    #racunanje trapeza zbog nejednakih momentata upetosti u cvorovima
        self.intrinsic_diagram_w_trap=self.intrinsic_diagram+np.ravel(trapezius)             #np.ravel funkcija potrebna da se mogu zbrojiti dva niza - jer su inace drugacijih oblika (R,1) i (R,)
        max_moment=np.max(self.intrinsic_diagram_w_trap)                #pronalazak maksimalnog momenta na gredi, starija verzija - max(self.intrinsic_diagram_w_trap) 
        self.max_s=abs(max_moment/self.prop.sect.Wy)                 #izracun maksimalnog naprezanja na gredi


class Structure():

    '''Supervising and super class that represents all of the model: it's nodes, sections, materials, properties, beams and loads assigned to beams. It's purpose is to establish
        communication between different objects and to run the analysis. It has access to all needed parts for optimization.'''

    def __init__(self,nodes,materials,sections,properties,beams):

        self.nodes:List[Node]=nodes

        self.materials:List[Material]=materials

        self.sections:List[Section]=sections

        print(properties)

        self.properties:List[Property]=properties

        self.beams:List[Beam]=beams

    def global_equation(self)-> np.ndarray: #vraca matricu nxn, gdje je n broj mogucih kuteva zakreta

        '''"Method that sets global system of linear algebraic equations for calculation of angular displacements based on compatibility conditions and equilibrium equations of every nodes.'''

        phi=np.zeros((len(self.nodes),1))  #inicijalizacija prazne matrice

        K=np.zeros((len(self.nodes),len(self.nodes)))
        m=np.zeros((len(self.nodes),1))
                                            #Petlja po numpyevim poljima je spora - Python je progr. jezik za prototipiranje.
        for beam in self.beams:             #OVO JE DOSLOVNO FORMIRANJE JEDNADŽBI RAVNOTEŽE ZA SVAKI CVOR! KORISTENJEM NUMPY-A IZVODENJE JE BRZE

                    ixgrid=np.ix_([beam.node1.ID-1, beam.node2.ID-1], [beam.node1.ID-1, beam.node2.ID-1])   #Koristenje np.ix_ da se na prava mjesta u globalnoj matrici doda lokalna krutost."
                    K[ixgrid]+=np.array([[2*beam.k_ij, -beam.k_ij], [-beam.k_ij, 2*beam.k_ij]])     #numpy zbrajanje preko submatrica  polje indeksa... np.ix_ numpy je brzi - da wrapped FORTRAN C++"

                    m[beam.node1.ID-1]+=(-beam.m12)      #VEKTOR MOMENATA
                    m[beam.node2.ID-1]+=(-beam.m21)


        K_inv = np.linalg.inv(K)                                                 #np.linalg.inv - funkcija iz np.linalg za racunanje inverza matrice.."
        uvjetovanost = np.linalg.norm(K,'fro')*np.linalg.norm(K_inv,'fro')       #ISPIS I RACUNANJE UVJETOVANOSTI RADI KONTROLE."
        
##        print("Uvjetovanost matrice K iznosi: \n", uvjetovanost)      #ovo oblikovati u metodu, koja provjerava numericku stranu, uvjetovanost matrice - to dodati u metodu koja daje status
##        print("Matrica K: \n", K)
##        print("Inverz matrice K: \n", K_inv)
##        print("Vektor momenata", m)


        phi = np.matmul(K_inv,m)       #matmul funkcija iz numpy-a za matricno mnozenje
        
        return phi



    def calculate_all(self):

        '''Method that based that calls function that calculates angular displacements, assigns those displacements to appropriate nodes and then calculates moments at the end of beams.'''

        phi=self.global_equation()

        #Pripisujemo cvorovima njihove kuteve zakreta

        for i in range(0,len(self.nodes)):
            self.nodes[i].phi=phi[i]        #POJASNJENJE: U cvoru, grede se zakrecu zajedno za isti kut. Primjetimo, ova metoda uzima u obzir krutosti. Tako da ce taj zakret biti
                                            #najblizi zakretu najkruce grede u stvarnosti. VAZNO - prema redoslijedu cvorova su i kreirane jednadzbe, pa tako ce i rjesenja biti ispravno poredana 


        #Momenti upetosti na kraju cvorova

        for beam in self.beams:
            beam.M12 = beam.k_ij*(2*beam.node1.phi-beam.node2.phi)+beam.m12
            beam.M21 = beam.k_ij*(2*beam.node2.phi-beam.node1.phi)+beam.m21
            beam.max_stress()

        #Maksimalno naprezanje grede okvira - #isto tako#

            #if self.beams[j].smax>self.beams[j].mat.sallow:
                #pozivati optimizacijski algoritam...
                #self.beams[j].     Tu slijedi neka promjena parametara pojedine grede tj. pojedinog sectiona. Jer, dimenzioniramo section.

        #Pozivanje optimizacijskog algoritma

            #VAZNO - iz structure moze se beamu dodijelit stress... ako on veci od dopustenog, mijenjaju se parametri beam-a. Ponovno iz structure-a metoda koja
            #Buduci da se optimizacijom mijenjaju neki propertyji, a neki ostaju, potrebno je svakoj gredi, dodijelit posebnu instancu propertya - dakle ne kao dosad! - tj. treba kreirat i novi section!
            #Za optimizacije materijala, treba jednostavno provjerit slucajeve i s drugim materijalima - mozda ugradit iskustvo unutra - npr. na visoko opterecnim ispod neke debljine ugradit posebni materijal - masa, cijena

    def change_opt_param(self):

        '''Method that changes parameters that is called after optimization step, calls for calculation of section parameters of beams and then calls for calculation of global equation again - Structure.calculate_all.
            Not yet implemented. There wasn't a need for it.'''

        pass

    def get_mass(self) -> float:

        '''Method that can be used by other programs in order to fetch (get) mass of a structure in order to form objective function'''

        mass=0
        for beam in structure_obj_beams:
            mass+=beam.length*beam.prop.sect.A*beam.prop.mat.dens

        return mass

class Input_file(): #DODATI NEKU FUNKCIONALNOST

    '''Class that stores data about input file used to generate analysis as well as functionality that goes with it.'''

    def __init__(self,path):
        self.path=path

class Constraint_Ratio(): #Kreiranje constraints moja ideja. Pomocu odredenih ulaznih parametara definira se i inicijalizira constraint kao objekt, a metoda se onda može iskoristiti kao funkcija da se oblikuje constraint.

    '''For now, only implemented for I beams. Otherwise, unpredictable results. Probably crash expected.'''

    def __init__(self,section:Section, species, which, type_o_inequality:str='GT', rhs:float=0 ):

        self.section=section
        self.rhs=float(self.section.dict_o_cons[species][which])
        self.type_o_ineq=type_o_inequality

        if species=='BF/TF':
            self.numerator=self.section.parameters[4]
            self.denominator=self.section.parameters[5]
        elif species=='HW/TW':
            self.numerator=self.section.parameters[0]
            self.denominator=self.section.parameters[1]
        elif species=='BF/HW':
            self.numerator=self.section.parameters[4]
            self.denominator=self.section.parameters[0]
        elif species=='TF/TW':
            self.numerator=self.section.parameters[5]
            self.denominator=self.section.parameters[1]
            
    def constraint(self,x) -> float:    #x je tu samo radi zahtjeva scipy funkcija

        if self.type_o_ineq=='GT':
            return self.numerator/self.denominator-self.rhs #inequality su non-negative tj. parameter -rhs > 0 , zato kad je parameter < rhs, treba preoblikovati u -parameter + rhs > 0
        else:
            return self.rhs-self.numerator/self.denominator
                

class Constraint_stress():

    def __init__(self,beam:Beam):

        self.beam=beam

    def constraint(self,x) -> float: #x je tu samo radi zahtjeva scipy funkcija

        sigmadop=self.beam.prop.mat.sigmadop
        max_stress=self.beam.max_s

        return sigmadop-max_stress

class Constraints_bounds():         #napraviti opciju za None u bounds-u... Ako je None, da ne kreira bounds za tu gornju ili donju granicu... 

    def __init__(self,section:Section ,tuple_in_bnds,ind_o_param):

        self.section=section
        self.ind_o_param=ind_o_param           # parametar za koji se kreiraju granice - gornja i donja
        self.lwbnd=tuple_in_bnds[0]                      #donja granica za parametar
        self.upbnd=tuple_in_bnds[1]                      #gornja granica za parametar

    def constraint_lower(self,parameters:List[float]) -> float:      

        return self.section.parameters[self.ind_o_param]-self.lwbnd    #ovaj puta nejde [ind_o_param-1] jer je brojac tako postavljen - vidi poziv Constraints_bounds u def cons_COBYLA
                                                                              #tu ce vjerojatno ici oni konektori
    def constraint_upper(self,parameters:List[float]) -> float:

        return self.upbnd-self.section.parameters[self.ind_o_param]
        
class Optimization():

    def __init__(self,sections_to_opt,num_o_param:int,optim_algorithm:str):

        self.parameters_0=[]
        self.num_o_param=num_o_param            #niz u koji ce se spremati za presjeke redom navedene za optimizaciju broj parametara potreban za promjenu presjeka..
        self.bnds=[]
        self.constraints_obj=[]
        self.mass=0

        for i in sections_to_opt:          #ova petlja rjesava dodjeljivanje pocetnog guess-a parameters_0
            
            self.parameters_0.append(structure_obj.sections[i-1].parameters)  #ovo zasad radi i svi parametri naznacenih greda predaju se u pocetne uvjete parameters_0

        if optim_algorithm=='SLSQP':
            
            self.cons=self.cons_SLSQP(sections_to_opt) #self.bonds unutar metode cons_SLSQP
            self.calc_opt_SLSQP(self.parameters_0,self.cons,self.bnds)
            
        elif optim_algorithm=='COBYLA':                          #COBYLA poznaje samo constraints, a ne poznaje bounds
            
            self.cons=self.cons_COBYLA(sections_to_opt)
            self.calc_opt_COBYLA(self.parameters_0,self.cons)


    def cons_SLSQP(self,sections_to_opt) -> tuple:

        constraints=[]

        for i in sections_to_opt:           #ova petlja rjesava dodijeljivanje bounds-a                                   #izbrisati samo ovaj redak i spojiti petlje
            
            if type(structure_obj.sections[i-1].bounds[0]) is tuple:    #nije bas najbolji nacin za ovo razlikovanje
                
                for j in structure_obj.sections[i-1].bounds:        
                    self.bnds.append(j)
                    
            else:
                self.bnds.append(structure_obj.sections[i-1].bounds)

        self.bnds=tuple(self.bnds)

        for beam in structure_obj.beams:   #kreiranje i definiranje ogranicenja (constraints) prema naprezanju za svaku gredu
            
            temp_obj=Constraint_stress(beam)
            self.constraints_obj.append(temp_obj)        #lista objekata constraints
            
            constraints.append({'type': 'ineq', 'fun': self.constraints_obj[-1].constraint}) #constraints za scipy.optimize.minimize

            
        constraints=self.ratio_constraints(constraints)

        return tuple(constraints)
    

    def cons_COBYLA(self,sections_to_opt) -> tuple:

        constraints=[]

        for i in sections_to_opt:

            ind_o_tuple=0
            
            for tuple_in_bnds in structure_obj.sections[i-1].bounds:     # tuple_in_bnds - tuple koji sadrzi jednu donju i jednu gornju granica za jedan parametar

                
                temp_obj=Constraints_bounds(structure_obj.sections[i-1], tuple_in_bnds, ind_o_tuple) #salje se greda da se ima pokazivac na zivi objekt, pa da se podaci osvjeze. ind_o_tuple - index tuple-a u tuple-u tuple-a.
                self.constraints_obj.append(temp_obj)

                constraints.append({'type': 'ineq', 'fun':self.constraints_obj[-1].constraint_lower})
                constraints.append({'type': 'ineq', 'fun':self.constraints_obj[-1].constraint_upper})

                ind_o_tuple+=1

        for i in structure_obj.beams:   #kreiranje i definiranje ogranicenja (constraints) za sve grede - sigma dop
            
            temp_obj=Constraint_stress(i)
            self.constraints_obj.append(temp_obj)        #lista objekata constraints
          
            constraints.append({'type': 'ineq', 'fun': self.constraints_obj[-1].constraint}) #constraints za scipy.optimize.minimize
            
        constraints=self.ratio_constraints(constraints)

        return tuple(constraints)

    def ratio_constraints(self,constraints): #izlaz je List razlicitih metodi constraints-ova

        '''New paragraph - constraint in form of ratios 1.6.2022. v0.33
        ------------------------------------------------'''
        for i in sections_to_opt:

            section=structure_obj.sections[i-1]
            
            for key in section.dict_o_cons: #sto ako dictionary ne postoji? vjerojatno samo preskace beam, i ne stvara constraints.. provjeriti.. 
                
                tuple_constr=section.dict_o_cons.get(key)
                min_value=tuple_constr[0]
                max_value=tuple_constr[1]

                which=0
                temp_obj=Constraint_Ratio(section, key, which,'GT' , min_value)
                self.constraints_obj.append(temp_obj)

                constraints.append({'type': 'ineq', 'fun': self.constraints_obj[-1].constraint}) 

                which=1
                temp_obj=Constraint_Ratio(section, key, which,'LT' , max_value)
                self.constraints_obj.append(temp_obj)

                constraints.append({'type': 'ineq', 'fun': self.constraints_obj[-1].constraint})

        return constraints

        '''------------------------------------------------'''
            
    def objfun(self,parameters):

        param_list=self.chunks(parameters, self.num_o_param)

        self.mass=0
        
        for i in sections_to_opt: #i je zapravo - ID odnosno redni broj section-a. Trebaju to dvoje biti istoznacni. 

            #section_index=sections_to_opt.index(i) #ovo nema smisla

            section_index=sections_to_opt.index(i)  #vraca indeks - koji kazuje za vrijednost i gdje i na kojem mjestu. 
            
            structure_obj.sections[i-1].calculate(param_list[section_index])

            for beam in structure_obj.beams:

                if beam.prop.sect.ID==i:
                    
                    beam.calculate_k_ij()
                    

        structure_obj.calculate_all()
        
        for i in structure_obj.beams:
            
            self.mass+=i.prop.sect.A*i.length       #zasad se masa racuna samo preko povrsine clanova - dakle to se nastoji minimizirati - lako se doda gustoca

        print('Masa: ' + str(self.mass))
            
        return self.mass

    def calc_opt_SLSQP(self,parameters_0,cons,bnds):

        print(cons)
        self.solution = sco.minimize(self.objfun, parameters_0, constraints=cons, bounds=bnds, method='SLSQP')

        print(self.solution.x)
        print(self.solution.success)
        print(self.solution.message)
        print('')
        print('Raw optimize.OptimizeResult: \n')
        print(self.solution)
        print('')

    def calc_opt_COBYLA(self,parameters_0,cons):

        self.solution = sco.minimize(self.objfun, parameters_0, constraints=cons,  method='COBYLA', options={'rhobeg': 10.0, 'maxiter': 2000})

        print(self.solution.x)
        print(self.solution.success)
        print(self.solution.message)
        print('')
        print('Raw optimize.OptimizeResult: \n')
        print(self.solution)
        print('')

    def chunks(self, parameters, num_o_param)->List[List]:

        stop=0
        output=[]
        for i in num_o_param:           #self.num_o_param niz treba utvrditi prema tipovima greda
            
            stop=stop+i
            output.append(parameters[stop - i:stop])
            
        return output   #izlaz je lista listi
    

def word_splitting(line_string:str)->List[str]:

    '''Function used for stripping and splitting words from textual file. Mainly used in creation of instances of a class.'''

    line_string=line_string.strip()     #cisti pocetak od razmaka

    line_list=line_string.split(",")    #razdvaja na pojedine rijeci

    i=0
    for word in line_list:
        word=word.strip()
        word=word.strip('"')
        line_list[i]=word
        i+=1
        
    return line_list

#CREATION FUNCTIONS - razne funkcije za kreiranje niza objekata na globalnoj razini...

def node_creation(line_list:List[str]):

    '''Function that creates NODE OBJECTS from input file.'''

    global numer_node, structure_obj

    node_name="Node"+str(numer_node)                                    #Automatsko dodavanje imena cvoru
    structure_obj.nodes.insert(numer_node,Node(node_name,line_list))    #KREIRANJE CVORA - staviti metodu append

    numer_node+=1                   #Povecavanje brojaca za 1


def material_creation(line_list):

    '''Function that creates MATERIAL OBJECTS from input file.'''

    global numer_mat, structure_obj
    global name_mat             #trenutno ovaj redak nije funkcionalan

    structure_obj.materials.insert(numer_mat,Material(line_list)) #pogledati datoteku sto je koji element niza line_list
    numer_mat+=1


def beam_creation(line_list):   #detaljnije pojasnjenje strukture - vidi u node_creation funkciji. Tamo je opisan pojedini redak - ovdje se ponavlja...

    '''Function that creates BEAM OBJECTS from input file.'''

    global numer_beam                                                             #trenutno ovaj jedan redak nije funkcionalan
    global name_beam, structure_obj


    beam_name="Beam"+str(name_beam)                                         # Vidjeti je li potrebno??? Moze bit, ali jedino ako se uz to omoguci unos kao dodatna mogucnost.
    structure_obj.beams.insert(numer_beam,Beam(beam_name,numer_beam,line_list))

    name_beam+=1
    numer_beam+=1


def section_type(line_list):

    '''Function that is actually a switch-case. Dependent on the read type of the section, it instatiates the right one'''

    chooser=line_list[1]

    global numer_sect, structure_obj

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

def section_creation(line_list):

    '''Function that creates SECTION OBJECTS from input file.'''

    global numer_sect, name_sect               #trenutno ovaj redak nije funkcionalan
    global lines, tpi

    section_type(line_list)
    numer_sect+=1


def property_creation(line_list):

    '''Function that creates PROPERTY OBJECTS from input file by assigning section and material to it. Also, takes parameter called "cost".
        That way, it is predicted that cost optimization is possible to achieve.'''

    global numer_prop

    structure_obj.properties.insert(numer_prop, Property(line_list)) #PROBLEM?
    numer_prop+=1


def load_creation(line_list):

    '''Function that calls proper beam object and creates loads that act on it'''

    global numer_load, structure_obj

    section_ID=int(line_list[2])
    structure_obj.beams[section_ID-1].create_load(line_list)


def switch_type(line_list):

    '''Function that calles appropriate function for creation of objects from input file.'''
    chooser=line_list[0]
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

def optim_creation(line_list):

    '''Function that stores data from text input file to objects connected with beams in order to create bounds and constraints upon instantiating optimization_obj'''

    section_ID=int(line_list[0])
    
    for section in structure_obj.sections:

        if section.ID == section_ID:
            
            if isinstance(section, I_Beam):
                
                section.optim(line_list[1:13])
                num_o_param.append(6)

            if isinstance(section, T_Beam_on_plate):
                
                section.optim(line_list[1:13])
                num_o_param.append(4)
            
            if isinstance(section, C_Beam):

                section.optim(line_list[1:9])
                num_o_param.append(4)

            if isinstance(section, CircBar):

                section.optim(line_list[1:3])
                num_o_param.append(1)

            if isinstance(section, CircTube):
                
                section.optim(line_list[1:5])
                num_o_param.append(2)

            if isinstance(section, Rectangle):

                section.optim(line_list[1:9])
                num_o_param.append(4)
                
        '''For creation of Ratio constraints - 1.6.2022. v0.33 '''
        for element in line_list:
            if element=='BF/TF':
                ind=line_list.index(element)
                section.ratio_constraint(line_list[ind:ind+3])              
            elif element=='HW/TW':
                ind=line_list.index(element)
                section.ratio_constraint(line_list[ind:ind+3])
            elif element=='BF/HW':
                ind=line_list.index(element)
                section.ratio_constraint(line_list[ind:ind+3])
            elif element=='TF/TW':
                ind=line_list.index(element)
                section.ratio_constraint(line_list[ind:ind+3])

def calculate_problem():

    '''Method used for optlib_pymoo.py for calling of calculation of already initialized problem. Problem is initialized by loading data from input text file.
    and is then connected via optlib_pymoo.py to a pymoo optimization module'''

    phi=structure_obj.global_equation()
    structure_obj.calculate_all()

def get_stress_cons_lt_0() -> List[float]:

    '''Method for pymoo and perhaps some other optimization modules that take list of values from evaluated constraints. For structural frame analysis it is common to compare beam stresses to allowable values for material.
        Results are returned in a list (pymoo) to be saved in "out" dictionary like this: out['G']=ao.get_stress_cons_lt_0() where ao is from: import Analiza_okvira as ao. '''

    beam_stress_cons=[]
    
    for beam in structure_obj.beams:
        con_value = beam.max_s - beam.mat.sigmadop
        beam_stress_cons.append(con_value)
        
        return beam_stress_cons

def get_stress_cons_gt_0() -> List[float]:

    beam_stress_cons=[]
    
    for beam in structure_obj.beams:
        con_value = beam.max_s - beam.mat.sigmadop
        beam_stress_cons.append(con_value)
        
        return beam_stress_cons

#       GlAVNI DIO PROGRAMA
#------------------------------------

#PUTANJA DO TEKSTUALNE ULAZNE DATOTEKE (BROWSE)


while 1:
    try:
        path=input("Upisati put do ulazne datoteke:")
        path=path.replace('\\','\\\\')
        print(path)
        inp_file=open(path,"rt")
    except:
        print("Greska. Ili datoteka ne postoji, ili je unesena putanja kriva.")
        print("Provjeriti ispravnost putanje i postojanje datoteke!")
    else: break

# UNOS KORAKA DISKRETIZACIJE! MOZE ICI IZ ULAZNE DATOTEKE!

while 1:
    try:
        kd=float(input("Unesite korak diskretizacije u jedinicama koordinatnog sustava: "))
        if kd<0:
            raise ValueError()
    except:
        print('Korak diskretizacije nije prikladan.')
    else: break
    
# ODABIR OPTIMIZACIJSKE METODE/OPTIMIZACIJSKOG ALGORITMA

while 1:
    try:
        optim_algorithm=str(input("Odabir optimizacijske metode/algoritma. Unesite 'C' za COBYLA ili 'S' za SLSQP: "))
        optim_algorithm=optim_algorithm.upper()
        if optim_algorithm=='S':
            optim_algorithm='SLSQP'
        elif optim_algorithm=='C':
            optim_algorithm='COBYLA'
            
    except:
        pass
    else: break


# DEFINICIJA NUKMERATORA ID-a ZA STVARANJE DEFAULTNIH IMENA AKO NE POSTOJE

numer_node=1
numer_mat=1
name_mat=1
numer_sect=1
name_sect=1
numer_prop=1
name_prop=1
numer_load=1
name_load=1
numer_beam=1
name_beam=1

#DEFINIRANJE POLJA OBJEKATA

nodes = []
materials = []
sections = []
properties = []
beams = []

structure_obj=Structure(nodes,materials,sections,properties,beams)

#DEFINIRANJE NEKIH GLOBALNIH BROJACA I LISTA

tpi=0                       #brojac koji broji kroz tekstualnu datoteku - tpi - text position index
lines= []                   #buduce linije tekstualne datoteke

#PROLAZENJE KROZ TEKSTUALNU DATOTEKU I ZATVARANJE DATOTEKE

for line_string in inp_file:
    lines.append(line_string)               #lines sada sadrze sav tekst tekstualne datoteke
inp_file.close()

#CITANJE BROJA OBJEKATA I KREIRANJE ISTIH POZIVANJEM FUNKCIJA

quantities_read=False
curr=0
num_o_objects=[]
sections_to_opt=[]
num_o_param=[]

while tpi<len(lines):

    if lines[tpi][0]!="#" and not quantities_read:  #u num_o_objects se pamti broj objekata - cvorova, presjeka, greda...
        quantities_read=True
        line_string=lines[tpi]                      #Prvih linija komentara moze biti vise, pa kad naide na liniju bez prvog znaka # - cita koliko ima kojih objekata
        line_string=line_string.strip()
        line_list=line_string.split(",")
        i=0
        for num in line_list:
            line_list[i]=int(num)
            i+=1
        num_o_objects=line_list

        #print(num_o_objects)                        #RADI KONTROLE

    elif lines[tpi][0]!="#" and quantities_read:

        if curr==0:                                 #NODE

            for i in range(0,num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                node_creation(line_list)
                tpi+=1

        elif curr==1:                                 #MATERIAL

            for i in range(0,num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                material_creation(line_list)
                tpi+=1

        elif curr==2:                                 #SECTION

            for i in range(0,num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                section_creation(line_list)
                tpi+=1

        elif curr==3:                                 #PROPERTY

            for i in range(0,num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                property_creation(line_list)
                tpi+=1

        elif curr==4:                                 #BEAM

            for i in range(0,num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                beam_creation(line_list)
                tpi+=1

        elif curr==5:                                 #LOAD

            for i in range(0,num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                load_creation(line_list)
                tpi+=1

        elif curr==6:                               #OPTIMIZATION

            for i in range(0, num_o_objects[curr]):

                line_string=lines[tpi]
                line_list=word_splitting(line_string)
                sections_to_opt.append(int(line_list[0]))        #u sections_to_opt se spremaju presjeci koje ce se optimizirati. Dakle, ne optimiziraju se grede vec presjeci!
                optim_creation(line_list)
                tpi+=1


        curr+=1

    tpi+=1




# KREIRANJE OBJEKTA KLASE STRUCTURE I ZAPOCINJANJE PRORACUNA


phi=structure_obj.global_equation()       #PROVJERENA UVJETOVANOST MATRICE ZA NEKE PROBLEME, I U REDU!
                                            #OVA METODA SE INACE POZIVA IZ CALCULATE ALL....
                                            #SVE BI OSTALO TREBALO BITI DOBRO - OSIM GLOBAL_EQUATION... TO PROVJERITI.
structure_obj.calculate_all()

print('Pocetna naprezanja: \n')

for i in structure_obj.beams:
    print(' ' + str(i.max_s))

print('Konacna naprezanja: \n')
for i in structure_obj.beams:
    print(' ' + str(i.max_s))

print('')
print('Konacni parametri: \n')

for i in structure_obj.beams:
    print(' Greda: ' + i.name)
    print(i.prop.sect.parameters)

optimization_obj=Optimization(sections_to_opt,num_o_param,optim_algorithm)

#za DEBUG i konacna vizuru rezultata

print('Konacna naprezanja: \n')
for i in structure_obj.beams:
    print(' ' + str(i.max_s))

print('')
print('Konacni parametri: \n')

for i in structure_obj.beams:
    print(' Greda: ' + i.name)
    print(i.prop.sect.parameters)

'''Za provjeru objekata - odlicna naredbe - object.__dict__ i isinstance(object,class)'''
