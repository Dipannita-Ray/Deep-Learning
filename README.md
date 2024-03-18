# Deep-Learning:
# LSTM:
Long Short Term Memory Netwok is a deep learning sequencial neural network that allows info to persist
It is a special type of recurrent neural network which was invented since RNN was unable to remember long term dependencies due to vanishing/exploding gradient problem .
it was designed by Hochreiter and schmidhuber hence resolving problems faced by rnn and ML algos
LSTM Model can be implemented in python using the keras library.
it is ideal for sequence prediction tasks.
## LSTM Archietecture:
![image](https://github.com/Dipannita-Ray/Deep-Learning/assets/155419369/b88fb391-b26b-4c57-a570-e77db9f82294)
here long term memory is konwn as cell state whereas short term memory is known a hidden state.LSTM also perfroms a communication between long and short term memory.
In general LSTM has three part [input ->processing -> output state] hence 
* input state has :
   1. previous cell states(t1,t0)
   2. previous hidden states
   3. input for the current time stamp.
    
* output state:
  1. current hidden state
  2. current cell state
* processing :
  1. update cell state (conversion of ct0 to ct1): here based on Xt which thing is to be forgeted is decided.And based on Xt which part is to be added is also based on current input. 
  2. calculation of ht
### Gates of LSTM : 
To update cell state and Calculate ht LSTM uses the concept of gates.
![image](https://github.com/Dipannita-Ray/Deep-Learning/assets/155419369/73e790d3-80c0-4bf4-b74d-d9945a653f6b)
* Forget gate : removes something that is not needed for the long term memory state.
* Input gate adds something
* calculate hidden state ht
* ht and ct are vectors hence dimension of both is same. Xt is also a vector .ft is present in forget gate. it and Ct bar is present in input gate and Ot is present in o/p gate . hence should be in same dimension.

* point wise Operation:
  1. * : ct-1 * ft
  2. + : ct-1 + ft
  3. tanh : tanh function is added to each vector points.
     
* neural network layers : these are yellow box in the diagram. hence are hyperparameter were no. of node is choice of our how many it can be. here sigmoid function is used to get a output between 0 and 1 whereas tanh gives o/p between -1 to 1.
* 
# Text generator  :
