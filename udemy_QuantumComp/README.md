## The Complete Quantum Computing Course
- Instructor: Atil Samancioglu

## Section 1: Introduction

1. Welcome

2. How to use this course?

3. Presentation to the course

## Section 2: Mathematical foundations

4. Intro to mathematical foundations

5. How clasical computers work?

6. Probability
- P(A): probability of A
- P(A AND B)
- P((A AND B) OR C)
- Mutually exclusive when P(A AND B) = 0
- Mutually exclusive when P(A OR B) = P(A) + P(B)
- Independent when P(A AND B) = P(A) * P(B)
- Independent when P(A OR B) = P(A) + P(B) - P(A AND B)

7. Statistics

8. Complex numbers

9. Matrix

10. Matrix operations

11. Special matrices

12. Linear transformation

## Section 3: Qubit and Physics

13. Qubit introduction

14. Superposition and interference

15. Entanglement

16. Qubit state
- [1 0]: spin down/qubit down
- [0 1]: spin up/qubit up
- Qubit: [p q] 
  - p = spin down probability
  - q = spin up probability
  - p^2 + q^2 = 1

17. Braket
- A^T*E = <A|E>
- |1> = [0 1]^T
- |0> = [1 0]^T
- |00> = [1 0 0 0]^T = |0><0|
- |01> = [0 1 0 0]^T = |0><1|
- |10> = [0 0 1 0]^T = |1><0|
- |11> = [0 0 0 1]^T = |1><1|

19. Multi Qubit

## Section 4: Python from scratch

## Section 5: Qiskit 101

64. Introduction to Qiskit

65. Classical Gates

66. IBM Signup

67. Quantum Gates

68. Entanglement
- Hadamard gate : $${1\over{\sqrt2}} \left[ \begin{array}{rr} 1 & 1 \\ 1 &-1 \end{array} \right] $$ 

69. Qiskit
- pip install qiskit

70. First Circuit

71. Running on Simulator
```
from qiskit import *
circuit = QuantumCircuit(2,2)
circuit.draw()
%matplotlib inline
circuit.draw(output='mpl')
circuit.h(0)
circuit.draw(output='mpl')
circuit.cx(0,1) # 0-> control qubit, 1-> target qubit
circuit.measure([0,1],[0,1])
circuit.draw(output='mpl')
simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit,backend=simulator).result()
from qiskit.visualization import plot_histogram
plot_histogram(result.get_counts(circuit))
```
- Run from jupyter notebook line by line

72. Getting Real Quantum Computer Properties

73. Running on Real Quantum Computer

74. Toffoli

75. GitHub Links
- https://github.com/atilsamancioglu/QX01-HelloQuantum
- https://github.com/atilsamancioglu/QX02-SimulatorsAndProviders

## Section 6: Teleportation

76. Introduction to Teleportation

77. Phase
- Z-gate: flip
  - Z =  $$ \left[ \begin{array}{rr} -1 & 0 \\ 0 & 1 \end{array} \right] $$ 
  - Z|0> = |0>
  - Z|1> = -|1>

78. Phase and Bloch sphere

79. Phase vs Bloch Sphere GitHub link

80. Superdense conding

81. Quantum Teleportation

82. teleportation in Qiskit

83. Quantum Teleportation GitHub Link
