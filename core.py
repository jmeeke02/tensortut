import tensorflow as tf

#TYPES OF NODES
#constants
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

print node1, node2

#Sessions
sess = tf.Session()
print(sess.run([node1, node2]))

#Operations
#combining tensor nodes with operations - which are also nodes

node3 = tf.add(node1, node2)
print "node3: ", node3
print "sess.run(node3): ", sess.run(node3)

#this graph always produces constant output

#Placeholders:
#this is the inputs, they are strictly typed
#parameterize graphs to accept external inputs = placeholders
#placeholder is a promise to provide a value later

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + operator is an alias for tf.add(a, b)

#like a function or a lambda - define 2 input paramters (a,b) and the operation on them
#we can evaluate this graph with multiple inputs usign feed_dict parameter 
    #lets us specify Tensors that provide concrete values to our Placeholders

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

#Make the computation graph more complex by adding another operation

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

#In Machine Learning:
# we want a model that can take arbitrary inputs - like the one above
# to make the model trainable:
    # need to be able to modify the graph 
    # get new outputs with the same input
    
    
#Variables:
#let us add trainable parameters to our graph
#constructed with a tyep and an initial value

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b 

#Initialization
#constants initialized when you call tf.constant 
    # values never change
#contrast:
    # not initialized when you call tf.Variable
    # to initialize all variables in a tensorflow program, need to call a special operation 

init = tf.global_variables_initializer()
sess.run(init)

#init
# a handle to the TensorFlow sub-graph
# intializes all the global Variables
# until we call sess.run - variables still uninitialized

# We can evaluate several values of x simultaneously using our linear model

print 'linear model', linear_model
print sess.run(linear_model, {x: [1,2,3,4]})


#MODEL
#Now we have a model, but not sure how good it is
#to evaluate the model on training data
    # need a y placeholder to provide desired values 
    # need to write a loss function

# Loss Function
# measures how far apart the curent mode is from the provided data
# standard loss model for linear regression
    # sums the squares of deltas(changes) between the current model and the provided data
    # linear_model - y 
        # the difference is how far off our linear model prediction was from teh desired result y
        # creates a vector where each element is the corresponding examples error delta
        # y  is the expected or desired value 
        # call tf.square to square that error
        #sum all the squares to create a single scalar using the reduce_sum function
        #tf.reduce_sum abstract the error of all examples using tf.reduce_sum
        
        
y = tf.placeholder(tf.float32)
square_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(square_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}))

#this produces loss value 23.66

#We can improve this maunally by reassinging the values W and b
    #give them the perfect values of -1 and 1
# A variable is initialized to the value provided to tf.Variable 
    #but thsi can be change with operations liek tf.assign
    #W = -1 and b=1 are the optimal paramters for our model 

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))

#0.0 is a perfect loss score
#we've guess the perfect values of W and b

#BUT the whole point of machine learning is to find the correct model paramters automatically
#this is what tf.trian does


#TRAIN
#TensorFlow provides optimizers
# Optimizers:
#     slowly change each variable in order to minmize the loss function 
#     many types of optimizers 
#     simplest optmizer = gradient descent
# 
# Gradient descent
#     modiefies each variable accoridn to magnitude of the derivative of loss with respect to that variable
#     computing symbolic derivatives manually is tedious & error prone
#     TensorFlow automatically produces derivatives given only
#         description of the model using the funciton tf.gradients
#         Optimizers sually do this for you

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) #reset variable value sot defaults, which ar eincorrect

for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    
print(sess.run([W, b]))
#OUTPUT: [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]

#results in the final model parameters
# both are highly close to -1 and 1

#This did actual machine Learning
#simple linear regression
    # doesn't require much Tensoflow core code
    # more complex models and methods willl need more code
#TensorFlow provides hgher level abstractions
    # ti make these more complicate modles and methods easier
    # abstractions for 
        #common patterns
        #strucutes
        #functionality
