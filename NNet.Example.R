#Declare the weights

w1 <- matrix(nrow = 4, ncol = 3, rnorm(mean = 0, sd = 1/(4*3), n = 4*3))
w2 <- matrix(nrow = 1, ncol = 4, rnorm(mean = 0, sd = 1/4, n = 4))

#Declare the bias

b1 <- matrix(nrow = 4, ncol = 1, 0)
b2 <- matrix(nrow = 1, ncol = 1, 0)

#Declare the input and target variable

x <- matrix(nrow = 3, ncol = 1, c(0.5, 0.2, -1.2))
y <- 0.987654321

#Declare the neurons

n1 <- matrix(nrow = 3, ncol = 1)
n2 <- matrix(nrow = 4, ncol = 1)
n3 <- matrix(nrow = 1, ncol = 1)

#Declare the sigmoid funtion

A <- function(x){return(1 / (1+exp(-x)))}

#Declare the derivative of the sigmoid function

A_prime <- function(x){return(-exp(-x) / (1 + exp(-x))^2)}

#Declare the loss function

L <- function(x,y){return((x-y)^2)}

#Declare the derivative of the loss function

L_prime <- function(x,y){return(2*(x-y))}

#Declare a step-size for backpropagation

t <- 0.005

#Loop the input throught the network

n1 <- A(x)

z2 <- w1 %*% n1 + b1
n2 <- A(z2)

z3 <-w2 %*% n2 + b2
n3 <- A(z3)

print(paste("output:", n3), quote = FALSE)

#Compute the sigmas (gradients with respect to z = Aw + b)

sigma1 <- L_prime(n3, y)
sigma2 <- sigma1 %*% w2 * t(A_prime(z2))

#Get the weight derivatives
#(It is computed as the outer product of the activated neurons and their z-gradients)

w1_deriv <- t(n1 %*% sigma2)
w2_deriv <- t(n2 %*% sigma1)

#Get the bias derivatives
#(They are the sigmas themselves)

b1_deriv <- t(sigma2)
b2_deriv <- sigma1

#Adjust the weights and bias

w1 <- w1 - t*w1_deriv
w2 <- w2 - t*w2_deriv

b1 <- b1 - t*b1_deriv
b2 <- b2 - t*b2_deriv


#Let's loop through the procedure a few times

results <- c()

repeat{

n1 <- A(x)

z2 <- w1 %*% n1 + b1
n2 <- A(z2)

z3 <-w2 %*% n2 + b2
n3 <- A(z3)

results <- c(results, n3)

if(abs(n3 - y) <= 0.001){break}

#Compute the sigmas (gradients with respect to z = Aw + b)

sigma1 <- L_prime(n3, y)
sigma2 <- sigma1 %*% w2 * t(A_prime(z2))

#Get the weight derivatives
#(It is computed as the outer product of the activated neurons and their z-gradients)

w1_deriv <- t(n1 %*% sigma2)
w2_deriv <- t(n2 %*% sigma1)

#Get the bias derivatives
#(They are the sigmas themselves)

b1_deriv <- t(sigma2)
b2_deriv <- sigma1

#Adjust the weights and bias

w1 <- w1 - t*w1_deriv
w2 <- w2 - t*w2_deriv

b1 <- b1 - t*b1_deriv
b2 <- b2 - t*b2_deriv

}

#Plot the absolute error

plot.frame <- matrix(nrow = length(results), ncol = 2)
plot.frame[,1] <- abs(results - y)
plot.frame[,2] <- c(1:length(results))

colnames(plot.frame) <- c("Error", "Iteration")

library(ggplot2)

ggplot(data = data.frame(plot.frame), aes(x = Iteration, y = Error)) + geom_line(color = "blue")


