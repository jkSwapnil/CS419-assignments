import numpy as np
 

def square_hinge_loss(targets, outputs):
  # Write thee square hinge loss here
  loss=0.0
  for i in range(len(outputs)):
    if(targets[i]==0):
      targets[i]=-1
    if(targets[i]*outputs[i]<1):
      loss+=(1-targets[i]*outputs[i])**2
  return loss
  #return 1.0

def logistic_loss(targets, outputs):
  # Write thee logistic loss loss here
  loss=0.0
  for i in range(len(outputs)):
    if(targets[i]==0):
      targets[i]=-1
    loss+=np.log(1+np.exp(-targets[i]*outputs[i]))
  return loss
  #return 1.0

def perceptron_loss(targets, outputs):
  # Write thee perceptron loss here
  loss=0.0
  for i in range(len(outputs)):
    if(targets[i]==0):
      targets[i]=-1
    if(targets[i]*outputs[i]<0):
      loss += -targets[i]*outputs[i]  
  return loss
  #return 1.0

def L2_regulariser(weights):
    # Write the L2 loss here
  Rw=0.0
  for i in range(len(weights)):
    Rw+=weights[i]**2
  return Rw
  #return 0.0

def L4_regulariser(weights):
    # Write the L4 loss here
  Rw=0.0
  for i in range(len(weights)):
    Rw+=weights[i]**4
  return Rw
  #return 0.0

def square_hinge_grad(weights,inputs, targets, outputs):
  # Write thee square hinge loss gradient here
  grad=np.zeros(len(weights), dtype=np.float32)

  for i in range(len(weights)):
    element=0
    for j in range(len(outputs)):
      if(targets[i]==0):
        targets[i]=-1
      if(targets[i]*outputs[i]<1):
        element = element + 2*((targets[j])**2)*outputs[j]*inputs[j][i] - 2*targets[j]*inputs[j][i]
    grad[i] = element
  return grad   
  #return np.random.random(11)

def logistic_grad(weights,inputs, targets, outputs):
  # Write thee logistic loss loss gradient here 
  grad=np.zeros(len(weights), dtype=np.float32)

  for i in range(len(weights)):
    element=0
    for j in range(len(outputs)):
      if(targets[i]==0):
        targets[i]=-1
      element = element - (targets[j]*inputs[j][i]*np.exp(-targets[i]*outputs[i]))/(1+np.exp(-targets[i]*outputs[i]))

    grad[i]=element
  return grad
  
  #return 1.00

def perceptron_grad(weights,inputs, targets, outputs):
  # Write thee perceptron loss gradient here
  grad=np.zeros(len(weights), dtype=np.float32)
  for i in range(len(weights)):
    element=0
    for j in range(len(outputs)):
      if(targets[i]==0):
        targets[i]=-1
      if(targets[i]*outputs[i]<0):
        element = element - targets[i]*inputs[j][i]
    grad[i] = element
  return grad
    #return np.random.random(11)

def L2_grad(weights):
  # Write the L2 loss gradient here
  grad=np.zeros(len(weights), dtype=np.float32)
  
  for i in range(len(weights)):
    grad[i]=2*weights[i]
  
  return grad  
  #return 0.00

def L4_grad(weights):
  # Write the L4 loss gradient here
  grad=np.zeros(len(weights), dtype=np.float32)
  
  for i in range(len(weights)):
    grad[i]=4*weights[i]**3
  
  return grad
  #return 0.00

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
                  "logistic_loss" : logistic_loss,
                  "perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
                       "logistic_loss" : logistic_grad,
                       "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
                              "L4" : L4_grad}
