#!/usr/bin/env python
# coding: utf-8

# In[31]:


## This code is created for QC mentorship program 2021 - screening TASK 1, Part 3  ##

# The code below implements a subTask #3 of TASK#1 to first create a N qubit register 'phi' with some random set of |0>'s 
# and |1>'s and the state of register is a product state 

# A quantum register 'psi' is then defnied and initialized to all |0>'s, In practice a control swap circuit on this two states 
# can be implemented in way gates are arranged as shown in step 1 qc.draw() HOWEVER THE MEASUREMENT on such a circuit will not
# be much helpful in inching towards the match (Maximise value of probability) as IN CASE OF NO PERFECT in two registers, it will 
# be representing orthogonal states and we will get similar value of probability (finding zero at aournd half of times)


## so to reconstruct N qubit state Phi in register Psi a qubit by qubit compare and reconstruction is implemented in next step

get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from iqx import *
import math 
from math import pi
import numpy as np

#provider = IBMQ.load_account()


# In[32]:


No_Qubits = 4  # Chose a value for number of Qubits. be considerate, we do not have Computer/simulator that can deal many 

q = QuantumRegister(1, 'control') # initialize a register for control qubit of size 1
c = ClassicalRegister(1) # initialize a register with control bit to read value from control qubit
qr_psi = QuantumRegister(No_Qubits, 'psi') # initialize a register to find the unknown quntum state phi
qr_phi = QuantumRegister(No_Qubits, 'phi') # initialize a register to store random qubit state we have to match with 

qc = QuantumCircuit(q, qr_psi, qr_phi, c) # design the cicuit with classical and quantum qubits 

# Get the random mix of  0 and 1's to initialize our N-qubit random state phi   
rotPhi = [[0, 0]] * No_Qubits
qbit_Phi = [0] * No_Qubits

for itr_Idx in range(No_Qubits) :
    qbit_Phi[itr_Idx] = np.random.randint(2)
    if qbit_Phi[itr_Idx] == 0 :
        rotvals = [0, 0]
    else :
        rotvals = [pi, 0]
    rotPhi[itr_Idx] = rotvals 

# print value of qubit phi and  obtain values corresponding rotations along Y and Z axis
print(' Qubit Phi', qbit_Phi, ' Rotations parameters/variables ', rotPhi)  

# start building N-qubits swaptest circuit
qc.h(0)  # perform haramard on 1st qubit 

# set psi qubit state to initial state of all |0>'s 
for itr_Idx in range(No_Qubits) :  
    qc.ry(0, qr_psi[itr_Idx])   # initialize psi qubits to |0>'s 
    qc.rz(0, qr_psi[itr_Idx])   
    qc.ry(rotPhi[itr_Idx][0], qr_phi[itr_Idx])   # initialize phi qubits with random mix of |0> and |1> 
    qc.rz(rotPhi[itr_Idx][1], qr_phi[itr_Idx])      
    qc.cswap(q, qr_psi[itr_Idx], qr_phi[itr_Idx]) # apply controlled swap from control qubit to psi and phi
    
qc.h(0)  # perform hadamard to change the basis back to computational basis 
qc.measure(q, c)  # perform measurement from control qubit to contorl bit 
qc.draw() 


# In[33]:


bkend = Aer.get_backend('qasm_simulator')
job = execute(qc, bkend, shots=1024)
counts = job.result().get_counts(qc)
ret_prob_0 = counts.get('0')/1024
print(ret_prob_0)


# In[36]:


# This circuit compares two qubits and return a cost function (probabiity) as a measure of match which approaches 
# to 1 in case of mathing qubits, it take only two parameters as two tuples

def getProb(v_prms, f_prms) :
    qr = QuantumRegister(1) # initialize a quantum register with a one control qubit
    cr = ClassicalRegister(1) # initialize a classical with one normal bit 
    w_qr = QuantumRegister(2) # initialize a two qubit for var and fix qubit

    yVar = v_prms[0]
    zVar = v_prms[1]
    yFix = f_prms[0]
    zFix = f_prms[1]
    
    var_qc = QuantumCircuit(qr, w_qr, cr) # create and quntum circuit object
    var_qc.h(0)

    var_qc.ry(yVar, w_qr[0])
    var_qc.rz(zVar, w_qr[0])
    var_qc.ry(yFix, w_qr[1])
    var_qc.rz(zFix, w_qr[1])
   
    var_qc.cswap (qr, w_qr[0], w_qr[1])
    var_qc.h(0)
    var_qc.measure(qr, cr)

    bkend = Aer.get_backend('qasm_simulator')
    job = execute(var_qc, bkend, shots=1024)
    counts = job.result().get_counts(var_qc)
    ret_prob_0 = counts.get('0')/1024
#    print('from routine prob and angles  ', ret_prob_0, yVar, zVar)
    return ret_prob_0


# In[54]:


## In below code we initialize the Psi qubit all |0>'s and Phi with a randon combination of |0>'s and |1>'s, then we a take
# qubits one by one and reconstruct the qubit in Psi regster using the help of getProb routine.  

probMax = 0.0
prob_Mx_AngleY = 0
prob_Mx_AngleZ = 0
old_Prob_0 = 0
sv_oldProb_0 = 0 
new_Prob_0 = 0
iter_Idx = 0
iter_Idx_Qcall = 0


vrotPsi = [[0, 0]] * No_Qubits  # populate the vparm with intial value of state |0>
qbit_Psi = [0] * No_Qubits
for itr_Idx in range(No_Qubits) :
    v_rotvals = [0, 0]
    vrotPsi[itr_Idx] = v_rotvals 

print(' The initial value of Register Psi: ', qbit_Psi, ' and the parms angles asso. wioth each qubit are: ', vrotPsi)  
print("\n Our goal is to reconstruct - ' The value in Register Phi: '", qbit_Phi)  

# Phi is a product state of  N qubits, in order to reconstruct product state of Phi in register Psi , a Qubit by Qubit
# SWAP test will be conducted in below code    

match_Flag = False 
for itr_Idx in range(No_Qubits) :  # Qubit by Qubit swap test
    v_rotvals = vrotPsi[itr_Idx]
    m_prob = getProb(vrotPsi[itr_Idx], rotPhi[itr_Idx]) 
    iter_Idx_Qcall += 1
    if m_prob < 1 :  # if two qubits are not aligned change the value of parms to state |1>   
        v_rotvals[0] = v_rotvals[0] + pi  # adjust angle parm that govern the rotation around  Y axis 
        m_newprob = getProb(v_rotvals, rotPhi[itr_Idx]) 
        iter_Idx_Qcall += 1
        if m_newprob == 1 :   # qubits are now aligned and in state |1> 
#            print(' FROM IF good match, prob. and parms' , m_newprob, parm_AngleY, parm_AngleZ)
            vrotPsi[itr_Idx] = v_rotvals
            match_Flag = True
            qbit_Psi[itr_Idx] = 1
        else :
            match_Flag = False
            print(' ERROR : exception, probabilities not computed correctly ')
    else :   
            vrotPsi[itr_Idx] = v_rotvals    # m_prob = 1 proving match with state |0>
            qbit_Psi[itr_Idx] = 0
            match_Flag = True

if (match_Flag) :
    print('\n -------- Reconstructed the', No_Qubits ,'Qubit product quantum state Phi in register Psi Successfully -------- ')    
    print('\nNumber of calls made to quantum routine        -- ', iter_Idx_Qcall)
    print('Value of register Phi Vs. Value of register Psi                  -- ', qbit_Phi, qbit_Psi)
    print('Value of params for individual qubits of phi                     --', vrotPsi)
    print('Value of params for individual qubits of Psi post reconstruction --', vrotPsi)
else :
    print(" Unable to find the Psi register asprobabilities are not computed correctly ")


# In[ ]:


### End of TASK1 Part3

