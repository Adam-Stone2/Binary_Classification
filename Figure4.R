# Rotated Axes Function
rotated_axes <- function(x, theta, delta) {
  u <- c(x[1] * cos(theta) - x[2] * sin(theta),
         x[1] * sin(theta) + x[2] * cos(theta))
  return(delta * (2 * (u[1] * u[2] > 0) - 1))
}

# Sinusoid Function
sinusoid <- function(x, theta, delta) {
  return(delta * (2 * (x[2] > theta * sin(10 * x[1])) - 1))
}

set.seed(14)  # For reproducibility
n <- 2000       # Number of points
x <- matrix(runif(2 * n, -1, 1), ncol = 2)  # Uniform points in [0, 1]^2
theta_rot <- c(0,pi/20,pi/14,pi/10,pi/6,pi/4)  # Example rotation angle for rotated axes
theta_sin <- c(0,0.1,0.3,0.5,0.75,1)    # Example amplitude for sinusoid
delta <- 4           # Jump size for the true function

Accuracy_values<-c("Addi_rotated","BART_rotated","Addi_sinusoidal","BART_sinusoidal")
Accuracy_values_rf<-c("Random_Forest_rotated","Random_Forest_rotated","Random_Forest_sinusoidal","Random_Forest_sinusoidal")

num_cores <- 10 # Specify the number of cores you want to use
cl <- makeCluster(num_cores)
registerDoParallel(cl)

Accuracy_results<-foreach(i = 1:6, .combine = rbind) %dopar%{ 
  
  library(randomForest)
  library(xgboost)
  #library(e1071)
  library(BayesTree)
  library('truncnorm')
  library('FNN')
  
  # Generate responses
  y_rotated <- ifelse((apply(x, 1, rotated_axes, theta = theta_rot[i], delta = delta) + rnorm(n, 0, 1))>0,1,0)
  #y_rotated_true <- apply(x, 1, rotated_axes, theta = theta_rot[i], delta = delta)

  
  TrainSet<-1:(4*n/8)
  TestSet<-(4*n/8):n
  Addi_results<-AddiVortes_Algorithm(y_rotated[TrainSet],x[TrainSet,],200,1200,400,1,0.4,1,25,y_rotated[TestSet],x[TestSet,])
  bart_results<-bart(x[TrainSet,],y_rotated[TrainSet],x[TestSet,])
  rf<-randomForest(x[TrainSet,],as.factor(y_rotated[TrainSet]),ntree=1000)
  xgb<-xgboost(data = xgb.DMatrix(data = x[TrainSet,], label = y_rotated[TrainSet]), nrounds = 1000,print_every_n = 100,objective = "binary:logistic")
  #svm<-svm(x[TrainSet,],y_rotated[TrainSet],probability = TRUE)  

  xgb_accuracy_rot<-sum(y_rotated[TestSet]==ifelse(predict(xgb,xgb.DMatrix(data = x[TestSet,])) > 0.5, 1, 0))/length(TestSet)
  bart_accuracy_rot<-sum(y_rotated[TestSet]==ifelse(colMeans(bart_results$yhat.test) > 0.5, 1, 0))/length(TestSet)
  random_forest_accuracy_rot<-sum(y_rotated[TestSet]==predict(rf,x[TestSet,]))/length(TestSet)
  
  
  return(c(Addi_results$Accuracy_test,bart_accuracy_rot,random_forest_accuracy_rot,xgb_accuracy_rot))
}

Accuracy_results_sin<-foreach(i = 1:6, .combine = rbind) %dopar%{ 
  
  library(randomForest)
  library(xgboost)
  #library(e1071)
  library(BayesTree)
  library('truncnorm')
  library('FNN')

  #y_sinusoidal_true <- apply(x, 1, sinusoid, theta = theta_sin[i], delta = delta) 
  
  y_sinusoidal <- ifelse(apply(x, 1, sinusoid, theta = theta_sin[i], delta = delta)>0,1,0)
  
  TrainSet<-1:(4*n/8)
  TestSet<-(4*n/8):n
  Addi_results_sin<-AddiVortes_Algorithm(y_sinusoidal[TrainSet],x[TrainSet,],200,1200,300,1,0.4,1,25,y_sinusoidal[TestSet],x[TestSet,])
  bart_results_sin<-bart(x[TrainSet,],as.factor(y_sinusoidal[TrainSet]),x[TestSet,])
  bart_accuracy_sin<-sum(y_sinusoidal[TestSet]==ifelse(colMeans(bart_results_sin$yhat.test) > 0.5, 1, 0))/length(TestSet)
  rf<-randomForest(x[TrainSet,],as.factor(y_sinusoidal[TrainSet]),ntree=1000)
  randome_forest_accuracy_sin<-sum(y_sinusoidal[TestSet]==predict(rf,x[TestSet,]))/length(TestSet)
  xgb<-xgboost(data = xgb.DMatrix(data = x[TrainSet,], label = y_sinusoidal[TrainSet]), nrounds = 1000,print_every_n = 100,objective = "binary:logistic")
  xgb_accuracy_sin<-sum(y_sinusoidal[TestSet]==ifelse(predict(xgb,xgb.DMatrix(data = x[TestSet,])) > 0.5, 1, 0))/length(TestSet)
  
  return(c(Addi_results_sin$Accuracy_test,bart_accuracy_sin,randome_forest_accuracy_sin, xgb_accuracy_sin))
}

stopCluster(cl)

par(mfrow=c(1,2))
par(mar = c(4, 4, 4, 4))

plot(theta_rot,as.numeric(Accuracy_results[,1]),ylim = c(0.87,1),pch=16,type="b",xlab="Rotation Angle",ylab="Accuracy")
points(theta_rot,as.numeric(Accuracy_results[,2]),col="red",pch=16,type = "b")
points(theta_rot,as.numeric(Accuracy_results[,3]),col="green",pch=16,type = "b")
points(theta_rot,as.numeric(Accuracy_results[,4]),col="blue",pch=16,type = "b")


#par(mfrow=c(1,2))
plot(theta_sin,as.numeric(Accuracy_results_sin[,1]),ylim = c(0.87,1),pch=16,type="b",xlab="Amplitude",ylab="Accuracy")
points(theta_sin,as.numeric(Accuracy_results_sin[,2]),col="red",pch=16,type = "b")
points(theta_sin,as.numeric(Accuracy_results_sin[,3]),col="green",pch=16,type = "b")
points(theta_sin,as.numeric(Accuracy_results_sin[,4]),col="blue",pch=16,type = "b")

