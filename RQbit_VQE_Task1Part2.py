#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This code is created for QC mentorship program 2021 - screening TASK 1, Part 2  ##

# The code below implements a subTask #2 of TASK#1 to first create a variational/parametric circuit to generate a random
# 1-qubit quantum state and run SWAP test to find the best choice and combination of two parameters such that it will 
# reproduce a randomly generated quantum state made in part 1.

# In the circuit 1st qubit is control, 2nd qubit is a work qubit that set to differnt states on the basis on two varying  
# parameters and third qubit is set to a fixed random value that work qubit will eventually chase in a vqe algorithm    

# This code resolves the part 2 of task with two differnt methods QSearch and GSearch which works on two different algorithms  
# to converge the cost towards the maximum value of '1'. A comparision of parms, iterations (calls to quanutm circtui routine) 
# and obtained results from these two ruotines are presented at the end of this program.

## Below are observation/notes for further enhancement of routines/algorithms 
##   The logic in algorithms can be enhanced further to overcome local minimums and fine tune states around the glob minimmm 
##   when identified 

## By analysing the algorithm with set of learning data, the Hyperparamters of algorithm can be tuned and we can learn the 
## most optimm values of the Constants/Model parameter (like, jump ratios, steps, graual increase parms in cae of local and
## global minimum etc.)  

## These two methods are subject to many enhancements but with one below limitation   -
## LIMITATION - The value of match/probability can be enhanced by increasing number of samples (circuit) we run in  getprob()
## routine however it may still return probilty value of '1' even if is not an absolute match between states but a close match, this 
## this can't be fixed by applying logic in classcial routine


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer, IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import math 
from math import pi
import numpy as np

# uncomment below line of code if quantum circuit has to be executed on real quantum computer 
#provider = IBMQ.load_account()


# In[2]:


# in order to accomplish part#2 of assignment TASK#1 it is requied to prepare and run a hybrid program composed of  classical 
# and qunatum parts, the step below shows how the quantum circuit looks like. In next few steps, variable parms are passed
# to a routine getprob() which inturn prepares gates  with these variational parms and run a quanutm circuit to calculate a 
# probabilitycost function with respect to chosen values of parms.

qr_c =  QuantumRegister(1, 'control') # Create and initializes a quantum register with one control qubit
cr_c =  ClassicalRegister(1, 'measurement') # initialize a classical register with one classical measurement bit

qr_var = QuantumRegister(1, 'chasing qubit') # quantum register to define chasing (variable) qubit
qr_fix = QuantumRegister(1, 'fix qubit') # quantum register to define fix qubit geneated in task#1

qc = QuantumCircuit(qr_c, qr_var, qr_fix, cr_c)

qc.h(0)  # perform haramard on 1st qubit

# initialize second (work) qubit to |0> state 
qc.ry(0, qr_var)
qc.rz(0, qr_var)

# prepare two random values to generate a random qubit state from a intial state of |0> 
rotY = np.random.random() * pi    # obtain random value for rotY to rotate state around Y axis between 0 to 180 degrees
rotZ = np.random.random() * 2*pi  # obtain a random values for rotZ to rotate state around Z axis between 0 to 360 degrees

#Let's initialize the third qubit with help of Random parmeters generated for TASK 1 subtask one
print(rotY, rotZ)

qc.ry(rotY, qr_fix)   # rotate the state of third qubit around Y by the random angle value rotY  
qc.rz(rotZ, qr_fix)   # rotate the state of third qubit around Z by the random angle value rotZ   

qc.draw()
qc.cswap(qr_c, qr_var, qr_fix)  # control cswap with control going from 1st qubit to work qubit and fix random qubit
qc.h(0) # perform haramard on first qubit to change the basis back to computational basis
qc.measure(qr_c[0], cr_c[0]) # perform measurement on 1 st qubit
qc.draw() # The quantum circuit used in program will look as shown below 


# In[3]:


# Display on bloch sphere the  third aubit in
cord_x = math.sin(rotY) * math.cos(rotZ)
cord_y = math.sin(rotY) * math.sin(rotZ)
cord_z = math.cos(rotY) 
print (' The values of cord_x, cord_y, cord_z on boch sphere ', cord_x, cord_y, cord_z, '\n')
plot_bloch_vector([cord_x, cord_y, cord_z])


# In[4]:


# define a function which runs quantum circuit and rerurns a value of cost in this case probability we want to maximize  
def getProb(yVar, zVar, yFix, zFix) :
    qr = QuantumRegister(1) # initialize a quantum register with a one control qubit
    cr = ClassicalRegister(1) # initialize a classical with one normal bit 
    w_qr = QuantumRegister(2) # initialize a two qubit for var and fix qubit

    var_qc = QuantumCircuit(qr, w_qr, cr) # create and quntum circuit object
    var_qc.h(0)
    ''' initialize 2nd qubit with fix parm we have to match with & first with varparms '''
    var_qc.ry(yVar, w_qr[0])
    var_qc.rz(zVar, w_qr[0])
    var_qc.ry(yFix, w_qr[1])
    var_qc.rz(zFix, w_qr[1])
   
    var_qc.cswap (qr, w_qr[0], w_qr[1])
    var_qc.h(0)
    var_qc.measure(qr, cr)

    ## checked circuit on real Quantum Computer ibmq_athens, since response is very slow, in rest of the program will be  
    ## using 'qasm_simulator' as device/backend to run quantum circuits.

    ##lb_device = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 3 and 
    ##                               not x.configuration().simulator and x.status().operational==True))
    ## print("Least busy device found is : ", lb_device)

    ##rqc_job = execute(qc, backend = lb_device, shots = 1024)
    ##job_monitor(rqc_job, interval = 2)
    ##rqc_result = rqc_job.result()
    ##counts = rqc_result.get_counts(qc)
    ##plot_histogram(counts)
    
    bkend = Aer.get_backend('qasm_simulator')
    job = execute(var_qc, bkend, shots=1024)
    counts = job.result().get_counts(var_qc)
    ret_prob_0 = counts.get('0')/1024
#    print('from routine prob and angles  ', ret_prob_0, yVar, zVar)
    return ret_prob_0


# In[5]:


# define a function to return an index associated with maximum value of probability parms passed to routine
def ret_maxval_idx(var0, var1, var2, var3) :
    Max_Prob = var0
    retVal = 0
    if var1 > Max_Prob :
        Max_Prob = var1
        retVal = 1
    if var2 > Max_Prob :
        Max_Prob = var2
        retVal = 2
    if var3 > Max_Prob :
        Max_Prob = var3
        retVal = 3
    
    print(' var s  ', var0, var1, var2, var3)
    print(' Max Prob ', Max_Prob, retVal)
    return retVal 


# In[6]:


# define a function to bring the variational parms back in range by assiging them an equivalent value when they are out of range
def conv_parms(v_prms) :
    prm_AngleY = v_prms[0]
    prm_AngleZ = v_prms[1]
    if prm_AngleY < 0 :
        prm_AngleY = prm_AngleY * -1
        prm_AngleZ = prm_AngleZ + pi
    if prm_AngleY > pi :
#        print ('***** check value ', prm_AngleY)
        prm_AngleY = prm_AngleY - (2*pi)
        if prm_AngleY < 0 :
            prm_AngleY = prm_AngleY * -1
            prm_AngleZ = prm_AngleZ + pi
        #  parm_AngleY = parm_AngleY * -1
        
    if prm_AngleZ < 0 :
        prm_AngleZ = 2*pi + prm_AngleZ 
    if prm_AngleZ > 2*pi :
        prm_AngleZ = prm_AngleZ / (2*pi)

    ret_prm = [prm_AngleY, prm_AngleZ]

#    print(' ret prams are  ', prm_AngleY, prm_AngleZ)
    return ret_prm 


# In[14]:


# Find state with method #1 QSearch using long jump, In this method a combinations of angleY and andgleZ are prepared (variable 
# parameters) at far and calls to quantum routine are made to determine the best cost. 

# Let's chose a initial set of parms an guess (Ansatz), parm angles of pi/2 and 0 are selected to create |+> state to start with
HLFY_ROT = pi/2
HLFZ_ROT = pi
parm_AngleY = HLFY_ROT   
parm_AngleZ = 0

print(" Values the parms for vector to reproduce ", rotY, rotZ)
print(" Initial values of parms for chasing vector ", parm_AngleY, parm_AngleZ)

probMax = 0.0
prob_Mx_AngleY = 0
prob_Mx_AngleZ = 0
old_Prob_0 = 0
sv_oldProb_0 = 0 
new_Prob_0 = 0
iter_Idx = 0
EXCP_STEP_INCR = 2*pi/200
STSFY99 = .99
GLBMAX_ENT = .995
m_AcceptedCost = 0.9999  # minimum satisfactory value to call it a match
m_Max_Iter = 25 # max iteration is set to 25 based on several test runs it can be adjusted based on the need 
iter_Idx_Qcall = 0
m_jump_Idx = 1
m_Max_jump_Idx = 10

old_Prob_0 = getProb(parm_AngleY, parm_AngleZ, rotY, rotZ)
iter_Idx_Qcall += 1
if old_Prob_0 >= m_AcceptedCost : 
    print("good match on choice of parms !! ")
else :
    step_fraction_idx = 0
    step_sizefrac = iter_Idx + 1 
    step_AngleY = pi/2     # initial step size for  variational parms 
    step_AngleZ = pi   
    
    # loop while you have good paramter values
    while (iter_Idx < m_Max_Iter and old_Prob_0 < m_AcceptedCost) :
        step_AngleY = pi/(2**(m_jump_Idx))     # set the new reduced steps for variational parms 
        step_AngleZ = pi/(2**(m_jump_Idx))
        m_jump_Idx += 1
        if m_jump_Idx > m_Max_jump_Idx and old_Prob_0 > STSFY99 : # reset the jumps 
            m_jump_Idx = 1
            
        cnv_prm = [parm_AngleY, parm_AngleZ]
        rslt_prm = conv_parms(cnv_prm)
        parm_AngleY = rslt_prm[0]
        parm_AngleZ = rslt_prm[1]

        ### call to get_probability value in -Y dimention
        new_Prob0_YD = getProb(parm_AngleY - step_AngleY, parm_AngleZ, rotY, rotZ)
        iter_Idx_Qcall += 1

        ### call to get_probability value in +Y dimention
        new_Prob0_YU = getProb(parm_AngleY + step_AngleY, parm_AngleZ, rotY, rotZ)
        iter_Idx_Qcall += 1

        ### call to get_probability value in -Z dimention
        new_Prob0_ZD = getProb(parm_AngleY, parm_AngleZ - step_AngleZ , rotY, rotZ)
        iter_Idx_Qcall += 1
   
        ### call to get_probability value in +Z dimention
        new_Prob0_ZU = getProb(parm_AngleY, parm_AngleZ + step_AngleZ , rotY, rotZ)
        iter_Idx_Qcall += 1
    
        valcase = ret_maxval_idx(new_Prob0_YD, new_Prob0_YU, new_Prob0_ZD, new_Prob0_ZU)  
        
        if valcase == 0 :
            parm_AngleY -= step_AngleY
            sv_oldProb_0 = new_Prob0_YD
        if valcase == 1 :
            parm_AngleY += step_AngleY
            sv_oldProb_0 = new_Prob0_YU
        if valcase == 2 :
            parm_AngleZ -= step_AngleZ
            sv_oldProb_0 = new_Prob0_ZD  
        if valcase == 3 :
            parm_AngleZ += step_AngleZ 
            sv_oldProb_0 = new_Prob0_ZU
        # check if we are at global maximum
        if (sv_oldProb_0 < old_Prob_0 and old_Prob_0 > GLBMAX_ENT) :
            excp_stepIncr = EXCP_STEP_INCR
            excp_stepSizeBack =  (EXCP_STEP_INCR) * -5
            excp_stepSizeZ = excp_stepSizeBack
            excp_Idx = 1
            new_Excp_Prob0 = getProb(parm_AngleY, parm_AngleZ + excp_stepSizeZ, rotY, rotZ)
            iter_Idx_Qcall += 1
            while (new_Excp_Prob0 < old_Prob_0 and old_Prob_0 > GLBMAX_ENT and excp_Idx < 10) :
                excp_stepSizeZ = excp_stepSizeBack + (excp_stepIncr * excp_Idx)  
                new_Excp_Prob0 = getProb(parm_AngleY, parm_AngleZ + excp_stepSizeZ, rotY, rotZ)
                iter_Idx_Qcall +=1
                excp_Idx += 1
                print(' new_Excp_Prob0 vs old_Prob_0 ', new_Excp_Prob0, old_Prob_0)
            if new_Excp_Prob0 > old_Prob_0 :
                parm_AngleZ = parm_AngleZ + excp_stepSizeZ
                old_Prob_0 = new_Excp_Prob0    
                print('parm_AngleZ, excp_stepSizeZ, new_Excp_Prob0, old_prob_0 ',parm_AngleZ, excp_stepSizeZ, new_Excp_Prob0, old_Prob_0)
        else :
            old_Prob_0 = sv_oldProb_0
   
        if old_Prob_0 > probMax :
            probMax = old_Prob_0
            prob_Mx_AngleY = parm_AngleY
            prob_Mx_AngleZ = parm_AngleZ
        if old_Prob_0 > m_AcceptedCost :
            break
        iter_Idx = iter_Idx + 1

cnv_prm = [parm_AngleY, parm_AngleZ]
rslt_prm = conv_parms(cnv_prm)
qs_parm_AngleY = rslt_prm[0]
qs_parm_AngleZ = rslt_prm[1]
qs_iter_Idx_Qcall = iter_Idx_Qcall

qs_final_prob = old_Prob_0

cnv_prm = [prob_Mx_AngleY, prob_Mx_AngleZ]
rslt_prm = conv_parms(cnv_prm)
prob_Mx_AngleY = rslt_prm[0]
prob_Mx_AngleZ = rslt_prm[1]

if probMax > old_Prob_0 :
    qs_parm_AngleY = prob_Mx_AngleY
    qs_parm_AngleZ = prob_Mx_AngleZ
    qs_final_prob = probMax

print(" Here are the final Probabilty and Variational angles from  QSearch routine :", old_Prob_0, qs_parm_AngleY, qs_parm_AngleZ)
print(" If number of steps are not enough a good approx of angles are ", probMax, prob_Mx_AngleY, prob_Mx_AngleZ )
print(" Number of calls made to to Quantum routine in this QSearch are ", qs_iter_Idx_Qcall)                            


# In[15]:


# find state with method #2, it looks or best value of parms in probability landscape and uses gradient to find 
# global maximum. it moves thru number of points to get the best choice of cost fucntion 

# chose a initial set of parms, let's choose parm angles pi/2 and 0 to create |+> state and start varying parms   
parm_AngleY = pi/2
parm_AngleZ = 0

print(" Values the parms for vector to reproduce ", rotY, rotZ)
print(" Initial values of parms for chasing vector ", parm_AngleY, parm_AngleZ)

probMax = 0.0
prob_Mx_AngleY = 0
prob_Mx_AngleZ = 0
old_Prob_0 = 0
new_Prob_0 = 0
iter_Idx = 0
itr_Mtch = 0
iter_Idx_Qcall = 0
GLBMAX_ENT = .99
EXCP_STEP_INCR = 2*pi/200
MAX_ITR = 40
MIN_ACT_COST = 0.9999  # minimum satisfactory value to call it a match
step_AngleY = pi/25     # initial step size for  variational parms 
step_AngleZ = 2*pi/50


old_Prob_0 = getProb(parm_AngleY, parm_AngleZ, rotY, rotZ)
iter_Idx_Qcall += 1
if old_Prob_0 >= MIN_ACT_COST : 
    print("good choice prob and parms ",old_Prob_0, parm_AngleY, parm_AngleZ)
else :
    print('step_AngleY ', step_AngleY)
    print('step_AngleZ ', step_AngleZ)
    step_fraction_idx = 0
    step_Angle_IncrY = step_AngleY
    step_Angle_IncrZ = step_AngleZ
    
    # loop while you have good paramter values
    while (iter_Idx < MAX_ITR and old_Prob_0 < MIN_ACT_COST) : 
        # if parms are out of range, bring it back to valid range in which we would like to measure results 
        cnv_prm = [parm_AngleY, parm_AngleZ]
        rslt_prm = conv_parms(cnv_prm)
        parm_AngleY = rslt_prm[0]
        parm_AngleZ = rslt_prm[1]

        # prepare parm to get cost wrt to one variable 
        n_parm_AngleY = parm_AngleY + step_AngleY 
        
        new_Prob0_AY = getProb(n_parm_AngleY, parm_AngleZ, rotY, rotZ)
        iter_Idx_Qcall += 1
        sv_parm_AngleY = n_parm_AngleY

        # check difference in cost and calculate gradient to decide move in desired direction 
        diff_Prob_AY = new_Prob0_AY - old_Prob_0
        grad_AngleY = diff_Prob_AY/step_AngleY

        # prepare parm to get cost wrt to another variable 
        n_parm_AngleZ = parm_AngleZ + step_AngleZ 
        
        new_Prob0_AZ = getProb(parm_AngleY, n_parm_AngleZ, rotY, rotZ)
        iter_Idx_Qcall += 1
        sv_parm_AngleZ = n_parm_AngleZ
        diff_Prob_AZ = new_Prob0_AZ - old_Prob_0
        
        grad_AngleZ = diff_Prob_AZ/step_AngleZ
  
        if (grad_AngleY >= 0) :
            n_parm_AngleY = parm_AngleY + (step_AngleY * 1)
        else :
            n_parm_AngleY = parm_AngleY - (step_AngleY * 1)
            
        if (grad_AngleZ > 0) :
            n_parm_AngleZ = parm_AngleZ + (step_AngleZ * 1)
        else :
            n_parm_AngleZ = parm_AngleZ - (step_AngleZ * 1)

        new_Prob0_YZ = getProb(n_parm_AngleY, n_parm_AngleZ, rotY, rotZ)
        iter_Idx_Qcall += 1
        
        valcase = ret_maxval_idx(old_Prob_0, new_Prob0_YZ, new_Prob0_AY, new_Prob0_AZ)  
        
        if valcase == 0 :          ##  handle non convergence 
            if step_Angle_IncrY > pi/10 :
                step_Angle_IncrY =  step_AngleY/2

            if step_Angle_IncrZ > pi/10 :
                step_Angle_IncrZ =  step_AngleZ/2
                
            step_AngleY = step_AngleY + step_Angle_IncrY 
            step_AngleZ = step_AngleZ + step_Angle_IncrZ 
            
            sv_oldProb_0 = old_Prob_0
            
        if valcase == 1 :
            parm_AngleY = n_parm_AngleY
            parm_AngleZ = n_parm_AngleZ
            sv_oldProb_0 = new_Prob0_YZ
        if valcase == 2 :
            parm_AngleY = sv_parm_AngleY
            sv_oldProb_0 = new_Prob0_AY
        if valcase == 3 :
            parm_AngleZ = sv_parm_AngleZ
            sv_oldProb_0 = new_Prob0_AZ

        if sv_oldProb_0 < 1 :
        # look for Global/Local Maximum
            if (sv_oldProb_0 <= old_Prob_0 and old_Prob_0 > GLBMAX_ENT and sv_oldProb_0 < 1) : # if we are in global max region
                excp_stepSizeIncr = EXCP_STEP_INCR
                excp_stepSizeBack =  (EXCP_STEP_INCR) * -5
                excp_Idx = 1
                excp_stepSizeZ = excp_stepSizeBack 
                new_Excp_Prob0 = getProb(parm_AngleY, parm_AngleZ + excp_stepSizeZ, rotY, rotZ)
                iter_Idx_Qcall += 1
                while (new_Excp_Prob0 < old_Prob_0 and old_Prob_0 > GLBMAX_ENT and excp_Idx < 10) :  # find global min  
                    excp_stepSizeZ = excp_stepSizeBack + excp_stepSizeIncr * (excp_Idx + 1)
                    new_Excp_Prob0 = getProb(parm_AngleY, parm_AngleZ + excp_stepSizeZ, rotY, rotZ)
                    excp_Idx += 1
                    iter_Idx_Qcall += 1
                if new_Excp_Prob0 > old_Prob_0 :
                    parm_AngleZ = parm_AngleZ + excp_stepSizeZ
                    sv_oldProb_0 = new_Excp_Prob0    
            
            if sv_oldProb_0 == old_Prob_0 : 
                itr_Mtch += 1
                print ('itr_Mtch', itr_Mtch)
                if (old_Prob_0 < GLBMAX_ENT) and (itr_Mtch > 3) : # Local maimum region
                    new_Mtch_ProbYPZM = getProb(parm_AngleY + (2 * step_Angle_IncrY), parm_AngleZ - (2 * step_Angle_IncrZ), rotY, rotZ)
                    new_Mtch_ProbYMZP = getProb(parm_AngleY - (2 * step_Angle_IncrY), parm_AngleZ + (2 * step_Angle_IncrZ), rotY, rotZ)
                    iter_Idx_Qcall += 2
                    if new_Mtch_ProbYPZM >= new_Mtch_ProbYMZP :
                        parm_AngleY = parm_AngleY + (2 * step_Angle_IncrY)
                        parm_AngleZ = parm_AngleZ - (2 * step_Angle_IncrZ)
                        old_Prob_0 = new_Mtch_ProbYPZM
                    else :
                        parm_AngleY = parm_AngleY - (2 * step_Angle_IncrY)
                        parm_AngleZ = parm_AngleZ + (2 * step_Angle_IncrZ)
                        old_Prob_0 = new_Mtch_ProbYMZP
                    itr_Mtch = 0
            else :
                  itr_Mtch = 0
            
            old_Prob_0 = sv_oldProb_0  # set value for next iteration
            
        ## END of code 

        if old_Prob_0 > probMax :
            probMax = old_Prob_0
            prob_Mx_AngleY = parm_AngleY
            prob_Mx_AngleZ = parm_AngleZ
        if old_Prob_0 > MIN_ACT_COST :
            print('good choice', old_Prob_0)
            break
        iter_Idx = iter_Idx + 1

cnv_prm = [parm_AngleY, parm_AngleZ]
rslt_prm = conv_parms(cnv_prm)
parm_AngleY = rslt_prm[0]
parm_AngleZ = rslt_prm[1]

gs_parm_AngleY = rslt_prm[0]
gs_parm_AngleZ = rslt_prm[1]
gs_iter_Idx_Qcall = iter_Idx_Qcall

gs_final_prob = old_Prob_0

cnv_prm = [prob_Mx_AngleY, prob_Mx_AngleZ]
rslt_prm = conv_parms(cnv_prm)
prob_Mx_AngleY = rslt_prm[0]
prob_Mx_AngleZ = rslt_prm[1]

if probMax > old_Prob_0 :
    gs_parm_AngleY = prob_Mx_AngleY
    gs_parm_AngleZ = prob_Mx_AngleZ
    gs_final_prob = probMax

print("Here are the final Probabilty and Variational angles from  GSearch routine :", old_Prob_0, gs_parm_AngleY, gs_parm_AngleZ)
print("If number of steps are not enough a good approx of angles are ", probMax, prob_Mx_AngleY, prob_Mx_AngleZ )
print("Number of calls made to to Quantum routine in this GSearch are ", gs_iter_Idx_Qcall)                            


# In[16]:


## The Comparision of results and expenses(iterations) from QSearch and GSearch
print(' ')
print(' Values of Parms for Qubit to recontruct        -- ', rotY, rotZ, '\n')
print('  ---------------------------- Results from QSearch Vs. GSearch --------------------- ')
print(' No. of iterations                              -', qs_iter_Idx_Qcall, ' Vs. ',  gs_iter_Idx_Qcall)
print(' Converged value of Probability                 -', qs_final_prob, ' Vs. ',  gs_final_prob)
print(' Final value of Parms after the reconstruction  -', qs_parm_AngleY, qs_parm_AngleZ, ' Vs.', gs_parm_AngleY, gs_parm_AngleZ)
                          


# In[17]:


print('\n Graphical Representation of Random Fixed Qubit with parms : ',  rotY, rotZ, '\n')
print(' Rotational Parms of Random Qubit State : ',  rotY, rotZ)
plot_bloch_vector([1, rotY, rotZ], title=" Qubit we wanted to reproduce ", coord_type='spherical')


# In[18]:


print('\n Rotational Parms of Qubit reproduced with QSearch :  ',  qs_parm_AngleY, qs_parm_AngleZ)
plot_bloch_vector([1, qs_parm_AngleY, qs_parm_AngleZ], title=" Qubit reproduced with QSearch", coord_type='spherical')


# In[19]:


print('\n Rotational Parms of Qubit reproduced  with  GSearch  ',  gs_parm_AngleY, gs_parm_AngleZ)
plot_bloch_vector([1, gs_parm_AngleY, gs_parm_AngleZ], title=" Qubit reproduced with GSearch", coord_type='spherical')


# In[13]:


### End of Part 2
## Below are observation/notes for further enhancement of routines/algorithms 
##   The logic in algorithms can be enhanced further to overcome local minimums and fine tune states around the glob minimmm 
##   when identified 

## By analysing the algorithm with set of learning data, the Hyperparamters of algorithm can be tuned and we can learn the 
## most optimm values of the Constants/Model parameter (like, jump ratios, steps, graual increase parms in cae of local and
## global minimum etc.)  

## These two methods are subject to many enhancements but with one below limitation   -
## LIMITATION - The value of match/probability can be enhanced by increasing number of samples (circuit) we run in  getprob()
## routine however it may still return probilty value of '1' even if is not an absolute match between states but a close match, this 
## this can't be fixed by applying logic in classcial routine

